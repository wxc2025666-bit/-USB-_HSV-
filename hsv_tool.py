"""电脑端 USB 摄像头 HSV 取色工具；不连接串口，不修改比赛程序。"""
import argparse
import copy
import datetime
import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from camera_worker import CameraWorker
from thresholds import (COLORS, DEFAULT_RANGES, controls_to_ranges,
                        ranges_to_controls, recommend_ranges, hsv_image,
                        threshold_mask, roi_samples, range_entry,
                        export_dictionary, config_from_json)


def demo_frame():
    """仅供无摄像头演示，不是现场阈值标定样本。"""
    frame = np.full((480, 640, 3), 205, np.uint8)
    swatches = [(0, 220, 205), (30, 225, 225), (115, 220, 150),
                (58, 205, 185), (0, 20, 35), (92, 100, 230)]
    for i, hsv in enumerate(swatches):
        color = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0]
        x, y = 120 + (i % 3) * 200, 145 + (i // 3) * 190
        cv2.ellipse(frame, (x + 8, y + 12), (65, 55), 0, 0, 360, (80, 80, 80), -1)
        cv2.circle(frame, (x, y), 58, tuple(int(v) for v in color), -1)
        cv2.putText(frame, str(i + 1), (x - 12, y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "DEMO - NOT CAMERA", (155, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
    return frame


class HSVTool:
    def __init__(self, root, demo=False):
        self.root = root
        self.root.title("USB摄像头 · 现场HSV阈值调试工具")
        width = min(1380, root.winfo_screenwidth() - 40)
        height = min(940, root.winfo_screenheight() - 80)
        root.geometry(f"{max(1000, width)}x{max(680, height)}")
        root.minsize(1000, 680)
        root.option_add("*Font", ("Microsoft YaHei UI", 9))
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TButton", padding=(6, 4))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        self.ranges = copy.deepcopy(DEFAULT_RANGES)
        self.samples = {i: [] for i in COLORS}
        self.color_id = 1
        self.worker = None
        self.image_label = None
        self.raw_frame = None
        self.display_frame = None
        self.frozen_frame = None
        self.frozen = False
        self.closed = False
        self.loading = False
        self.valid = True
        self.roi = None
        self.drag_start = None
        self.point = None
        self.transforms = {}
        self.photos = {}
        self.index = tk.IntVar(value=1)
        self.backend = tk.StringVar(value="DSHOW")
        self.resolution = tk.StringVar(value="640x480")
        self.color_name = tk.StringVar(value="1  红(red)")
        self.blur = tk.BooleanVar(value=True)
        self.clean = tk.BooleanVar(value=True)
        self.auto_exposure = tk.BooleanVar(value=True)
        self.exposure = tk.IntVar(value=-6)
        self.coverage = tk.StringVar(value="98")
        self.pads = [tk.IntVar(value=v) for v in (4, 15, 15)]
        self.values = [tk.IntVar(value=v) for v in ranges_to_controls(self.ranges[1])]
        self.notice = tk.StringVar(value="先打开外接USB摄像头，选择颜色，在物料顶面拖框取色。不要框入背景。")
        self.camera_status = tk.StringVar(value="未打开摄像头。外接USB常为编号1或2；编号0可能是电脑内置摄像头。")
        self.pixel_status = tk.StringVar(value="鼠标指向原图可读 RGB / HSV；拖框会暂停画面并自动推荐阈值。")
        self.sample_status = tk.StringVar()
        self.cover_status = tk.StringVar(value="白色 = 当前阈值通过；黑色 = 被排除。")
        self._build()
        for value in self.values:
            value.trace_add("write", self._on_values)
        self._update_code()
        self._update_samples()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda event: self.close())
        if demo:
            self.set_image(demo_frame(), "演示图（并非摄像头现场样本）")
        self.after_id = root.after(50, self._tick)

    def _build(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(top, text="USB设备编号").pack(side="left")
        ttk.Spinbox(top, from_=0, to=9, width=3, textvariable=self.index).pack(side="left", padx=5)
        ttk.Combobox(top, values=["DSHOW", "MSMF"], textvariable=self.backend,
                     state="readonly", width=7).pack(side="left", padx=3)
        ttk.Combobox(top, values=["640x480", "1280x720"], textvariable=self.resolution,
                     state="readonly", width=10).pack(side="left", padx=3)
        ttk.Button(top, text="打开摄像头", command=self.open_camera).pack(side="left", padx=3)
        ttk.Button(top, text="关闭摄像头", command=self.stop_camera).pack(side="left", padx=3)
        self.pause_button = ttk.Button(top, text="暂停 / 取色", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=3)
        ttk.Button(top, text="打开图片", command=self.load_image).pack(side="left", padx=3)
        ttk.Button(top, text="演示图", command=lambda: self.set_image(demo_frame(), "演示图（并非现场样本）")).pack(side="left", padx=3)
        exposure_row = ttk.Frame(main)
        exposure_row.pack(fill="x")
        ttk.Checkbutton(exposure_row, text="自动曝光", variable=self.auto_exposure).pack(side="left")
        ttk.Label(exposure_row, text="手动曝光值（依驱动）").pack(side="left", padx=(8, 2))
        ttk.Spinbox(exposure_row, from_=-13, to=0, width=4, textvariable=self.exposure).pack(side="left", padx=3)
        ttk.Button(exposure_row, text="提交曝光设置", command=self.apply_exposure).pack(side="left", padx=6)
        ttk.Checkbutton(exposure_row, text="5×5模糊（与main5取色一致）", variable=self.blur,
                        command=self.preprocessing_changed).pack(side="left", padx=8)
        ttk.Checkbutton(exposure_row, text="掩膜去噪（开闭运算）", variable=self.clean).pack(side="left", padx=3)
        ttk.Label(main, textvariable=self.camera_status, foreground="#526171").pack(fill="x", pady=(5, 7))
        body = ttk.Frame(main)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)
        views = ttk.Frame(body)
        views.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        views.columnconfigure(0, weight=1)
        views.columnconfigure(1, weight=1)
        views.rowconfigure(1, weight=3)
        views.rowconfigure(4, weight=2)
        ttk.Label(views, text="原图：拖框取色 · 移动鼠标读数", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        self.original = tk.Canvas(views, background="#18212b", highlightthickness=0, width=640, height=330)
        self.original.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=4)
        self.original.bind("<ButtonPress-1>", self.mouse_down)
        self.original.bind("<B1-Motion>", self.mouse_drag)
        self.original.bind("<ButtonRelease-1>", self.mouse_up)
        self.original.bind("<Motion>", self.mouse_move)
        ttk.Label(views, textvariable=self.pixel_status, wraplength=760).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(views, text="二值掩膜（白色为选中）").grid(row=3, column=0, sticky="w")
        ttk.Label(views, text="筛选结果（检查误选与漏选）").grid(row=3, column=1, sticky="w", padx=6)
        self.mask_canvas = tk.Canvas(views, background="#18212b", highlightthickness=0, width=320, height=220)
        self.mask_canvas.grid(row=4, column=0, sticky="nsew", pady=4, padx=(0, 3))
        self.result_canvas = tk.Canvas(views, background="#18212b", highlightthickness=0, width=320, height=220)
        self.result_canvas.grid(row=4, column=1, sticky="nsew", pady=4, padx=(3, 0))
        ttk.Label(views, textvariable=self.cover_status, wraplength=760).grid(row=5, column=0, columnspan=2, sticky="w", pady=4)
        # 右栏可滚动，低分辨率笔记本屏幕仍能找到导出按钮。
        right = ttk.Frame(body, width=375)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        scroll = tk.Canvas(right, highlightthickness=0, width=350)
        scroll.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=scroll.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        scroll.configure(yscrollcommand=scrollbar.set)
        panel = ttk.Frame(scroll, padding=(5, 0, 10, 6))
        panel_window = scroll.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda event: scroll.configure(scrollregion=scroll.bbox("all")))
        scroll.bind("<Configure>", lambda event: scroll.itemconfigure(panel_window, width=event.width))
        def wheel(event):
            scroll.yview_scroll(-int(event.delta / 120), "units")
        # 仅在右栏滚动，左侧预览不受影响。
        self.root.bind("<MouseWheel>", lambda event: wheel(event) if str(event.widget).startswith(str(panel)) or event.widget == scroll else None)
        ttk.Label(panel, text="1. 选择要标定的物料", style="Title.TLabel").pack(anchor="w")
        combo = ttk.Combobox(panel, textvariable=self.color_name, state="readonly",
                             values=[f"{i}  {name}" for i, name in COLORS.items()])
        combo.pack(fill="x", pady=5)
        combo.bind("<<ComboboxSelected>>", self.change_color)
        ttk.Label(panel, text="2. 拖框推荐，再微调上下限", style="Title.TLabel").pack(anchor="w", pady=(7, 3))
        ttk.Label(panel, text="OpenCV：H 0~179；S、V 0~255。\nH下限大于上限 = 跨179/0的两段范围。", wraplength=335).pack(anchor="w")
        for i, (name, maximum) in enumerate(zip(("H 色相下限", "H 色相上限", "S 饱和下限", "S 饱和上限", "V 明度下限", "V 明度上限"), (179, 179, 255, 255, 255, 255))):
            row = ttk.Frame(panel)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=name, width=11).pack(side="left")
            tk.Scale(row, variable=self.values[i], from_=0, to=maximum,
                     orient="horizontal", showvalue=False, resolution=1,
                     highlightthickness=0, length=150, sliderlength=16).pack(side="left", fill="x", expand=True)
            ttk.Spinbox(row, from_=0, to=maximum, width=4, textvariable=self.values[i]).pack(side="left", padx=(4, 0))
        sample_box = ttk.LabelFrame(panel, text="推荐参数 / 多次采样", padding=5)
        sample_box.pack(fill="x", pady=7)
        row = ttk.Frame(sample_box)
        row.pack(fill="x")
        ttk.Label(row, text="样本覆盖率 %").pack(side="left")
        ttk.Combobox(row, textvariable=self.coverage, values=["90", "95", "98", "100"], width=4).pack(side="left", padx=4)
        ttk.Label(row, text="默认98，过滤少量离群点").pack(side="left")
        row = ttk.Frame(sample_box)
        row.pack(fill="x", pady=4)
        for label, value, maximum in zip(("H余量", "S余量", "V余量"), self.pads, (30, 100, 100)):
            ttk.Label(row, text=label).pack(side="left")
            ttk.Spinbox(row, from_=0, to=maximum, width=3, textvariable=value).pack(side="left", padx=(2, 5))
        ttk.Label(sample_box, textvariable=self.sample_status).pack(anchor="w")
        row = ttk.Frame(sample_box)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="用样本重新推荐", command=self.recommend).pack(side="left")
        ttk.Button(row, text="清空当前样本", command=self.clear_samples).pack(side="left", padx=4)
        ttk.Label(sample_box, text="每次拖框自动加入样本（最多10次）。\n换光照后继续框选；换颜色先切换上方选项。", wraplength=320).pack(anchor="w")
        ttk.Label(panel, textvariable=self.notice, foreground="#97511b", wraplength=335).pack(fill="x", pady=(3, 7))
        ttk.Label(panel, text="3. 可直接填入 main5 的字典项", style="Title.TLabel").pack(anchor="w")
        self.code = tk.Text(panel, height=4, wrap="word", font=("Consolas", 10),
                            background="#ecf3f8", relief="flat", padx=6, pady=6)
        self.code.pack(fill="x", pady=5)
        row = ttk.Frame(panel)
        row.pack(fill="x")
        ttk.Button(row, text="复制当前颜色", command=self.copy_current).pack(side="left")
        ttk.Button(row, text="复制全部6色", command=self.copy_all).pack(side="left", padx=5)
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=6)
        ttk.Button(row, text="导出阈值 TXT", command=self.export_text).pack(side="left")
        ttk.Button(row, text="恢复当前默认值", command=self.restore_default).pack(side="left", padx=5)
        row = ttk.Frame(panel)
        row.pack(fill="x")
        ttk.Button(row, text="保存配置 JSON", command=self.save_config).pack(side="left")
        ttk.Button(row, text="加载配置 JSON", command=self.load_config).pack(side="left", padx=5)
        ttk.Label(panel, text="仅帮助HSV色域调试，不修改YOLO模型。\n黑色仍需ROI/形状过滤；阈值过宽会选中阴影。\n未采样的颜色仍保留默认值，导出前逐色验证。", wraplength=335, foreground="#526171").pack(anchor="w", pady=(9, 3))
        ttk.Label(main, text="流程：打开USB → 选择颜色 → 框选物料顶面 → 看掩膜/调滑块 → 恢复实时验证 → 复制或导出。Esc退出。", foreground="#526171").pack(fill="x", pady=(7, 0))

    def _on_values(self, *args):
        if self.loading:
            return
        try:
            self.ranges[self.color_id] = controls_to_ranges([value.get() for value in self.values])
            self.valid = True
            self._update_code()
        except (ValueError, tk.TclError) as exc:
            self.valid = False
            self._set_code("参数无效：S、V下限不能超过上限；所有值需在合法范围。\n" + str(exc))

    def _set_values(self, ranges):
        self.loading = True
        try:
            for var, value in zip(self.values, ranges_to_controls(ranges)):
                var.set(value)
        finally:
            self.loading = False
        self.ranges[self.color_id] = ranges
        self.valid = True
        self._update_code()

    def _set_code(self, text):
        self.code.configure(state="normal")
        self.code.delete("1.0", "end")
        self.code.insert("1.0", text)
        self.code.configure(state="disabled")

    def _update_code(self):
        self._set_code(range_entry(self.color_id, self.ranges[self.color_id]))

    def _update_samples(self):
        groups = self.samples[self.color_id]
        self.sample_status.set(f"当前颜色：{len(groups)}次采样，{sum(len(x) for x in groups)}个像素")

    def change_color(self, event=None):
        self.color_id = int(self.color_name.get().split()[0])
        self._set_values(self.ranges[self.color_id])
        self.roi = None
        self.point = None
        self._update_samples()
        self.notice.set("已切换颜色。请只框选这一种物料，不要把多个颜色一起框入。")

    def restore_default(self):
        self._set_values(copy.deepcopy(DEFAULT_RANGES[self.color_id]))
        self.notice.set("已恢复当前颜色默认范围（不代表现场最佳范围）。样本仍保留。")

    def clear_samples(self):
        self.samples[self.color_id].clear()
        self.roi = None
        self._update_samples()
        self.notice.set("已清空当前颜色的取色样本，阈值未改动。")

    def preprocessing_changed(self):
        for groups in self.samples.values():
            groups.clear()
        self._update_samples()
        self.notice.set("预处理已改变，旧样本已清空；请重新框选。与main5一致时保持5×5模糊开启。")

    def recommend(self):
        groups = self.samples[self.color_id]
        if not groups:
            self.notice.set("还没有样本，请在物料顶面拖框。")
            return
        try:
            ranges, warnings = recommend_ranges(np.concatenate(groups), self.color_id,
                float(self.coverage.get()), *(value.get() for value in self.pads))
            self._set_values(ranges)
            text = "已按多次样本推荐。请恢复实时画面，验证整块物料和背景。"
            if warnings:
                text += "\n" + "\n".join(warnings)
            self.notice.set(text)
        except (ValueError, tk.TclError) as exc:
            self.notice.set(str(exc))

    def open_camera(self):
        if self.worker and self.worker.thread.is_alive():
            self.notice.set("请先关闭当前摄像头，等状态显示已关闭后再打开。")
            return
        try:
            index = self.index.get()
            if not 0 <= index <= 9:
                raise ValueError("摄像头编号应为0~9")
            size = tuple(map(int, self.resolution.get().split("x")))
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("设备参数", str(exc), parent=self.root)
            return
        self.raw_frame = self.display_frame = self.frozen_frame = None
        self.image_label = None
        self.frozen = False
        self.roi = self.point = None
        self.pause_button.configure(text="暂停 / 取色")
        self.worker = CameraWorker(index, self.backend.get(), size)
        self.camera_status.set("正在打开摄像头…")
        self.worker.start()

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            if not self.frozen and self.display_frame is not None:
                self.frozen = True
                self.frozen_frame = self.display_frame.copy()
                self.pause_button.configure(text="恢复实时 / 继续验证")
            self.camera_status.set("正在关闭摄像头，等待驱动释放…")
        else:
            self.camera_status.set("摄像头已关闭。")

    def apply_exposure(self):
        if not self.worker or not self.worker.thread.is_alive() or self.worker.stop_event.is_set():
            self.notice.set("先打开摄像头，再提交曝光设置。")
            return
        try:
            self.worker.set_exposure(self.auto_exposure.get(), self.exposure.get())
            self.notice.set("已请求设置曝光；有些USB驱动不支持。预览确认变化后重新采样，保持与车上设置一致。")
        except (ValueError, tk.TclError) as exc:
            self.notice.set(str(exc))

    def set_image(self, frame, label):
        self.stop_camera()
        self.raw_frame = frame.copy()
        self.frozen_frame = frame.copy()
        self.display_frame = frame.copy()
        self.frozen = True
        self.roi = self.point = None
        self.image_label = label + f" · {frame.shape[1]}×{frame.shape[0]}"
        self.pause_button.configure(text="恢复实时 / 继续验证")
        self.camera_status.set(self.image_label)
        self.notice.set("当前是静态图片，可拖框取色。现场标定请使用实际USB摄像头。")

    def load_image(self):
        path = filedialog.askopenfilename(parent=self.root, title="打开取色图片",
                filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("全部文件", "*.*")])
        if not path:
            return
        try:
            # Windows中文路径不能交给旧版imread直接处理。
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("不是可读取的图片")
            self.set_image(frame, "图片：" + os.path.basename(path))
        except Exception as exc:
            messagebox.showerror("打开图片失败", str(exc), parent=self.root)

    def toggle_pause(self):
        if self.frozen:
            if not self.worker or not self.worker.thread.is_alive() or self.worker.stop_event.is_set():
                self.notice.set("当前没有实时摄像头；先打开USB摄像头，才能继续实时验证。")
                return
            self.frozen = False
            self.frozen_frame = None
            self.pause_button.configure(text="暂停 / 取色")
        elif self.display_frame is not None:
            self.frozen = True
            self.frozen_frame = self.display_frame.copy()
            self.pause_button.configure(text="恢复实时 / 继续验证")

    def _image_xy(self, event):
        transform = self.transforms.get(self.original)
        if transform is None:
            return None
        left, top, scale_x, scale_y, width, height = transform
        x, y = (event.x - left) / scale_x, (event.y - top) / scale_y
        if not (0 <= x < width and 0 <= y < height):
            return None
        return int(x), int(y)

    def mouse_down(self, event):
        point = self._image_xy(event)
        if point is None or self.display_frame is None:
            return
        self.frozen = True
        self.frozen_frame = self.display_frame.copy()
        self.pause_button.configure(text="恢复实时 / 继续验证")
        self.drag_start = point
        self.point = point
        self.roi = None

    def mouse_drag(self, event):
        point = self._image_xy(event)
        if self.drag_start is None or point is None:
            return
        x, y = self.drag_start
        px, py = point
        self.roi = (min(x, px), min(y, py), max(x, px) + 1, max(y, py) + 1)

    def mouse_up(self, event):
        if self.drag_start is None:
            return
        self.mouse_drag(event)
        self.drag_start = None
        if self.roi and self.frozen_frame is not None:
            try:
                batch = roi_samples(hsv_image(self.frozen_frame, self.blur.get()), self.roi)
                groups = self.samples[self.color_id]
                groups.append(batch)
                del groups[:-10]
                self._update_samples()
                self.recommend()
            except ValueError:
                self.roi = None
                self.notice.set("点击可看单点HSV；要自动推荐请拖一个至少4×4像素的框。")

    def mouse_move(self, event):
        point = self._image_xy(event)
        frame = self.display_frame
        if point is None or frame is None:
            return
        x, y = point
        b, g, r = map(int, frame[y, x])
        # 单点读数采用与取色相同的预处理，避免模糊开启时读数和阈值不一致。
        h, s, v = map(int, hsv_image(frame, self.blur.get())[y, x])
        self.pixel_status.set(f"像素 ({x},{y})  RGB=({r},{g},{b})  HSV=({h},{s},{v})" + ("  [5×5模糊后HSV]" if self.blur.get() else ""))

    def _draw(self, canvas, pixels, key, rgb=False):
        image = Image.fromarray(pixels if rgb or pixels.ndim == 2 else cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
        cw, ch = max(2, canvas.winfo_width()), max(2, canvas.winfo_height())
        width, height = image.size
        scale = min(cw / width, ch / height)
        dw, dh = max(1, int(width * scale)), max(1, int(height * scale))
        image = image.resize((dw, dh), Image.Resampling.BILINEAR)
        self.photos[key] = ImageTk.PhotoImage(image)
        left, top = (cw - dw) // 2, (ch - dh) // 2
        canvas.delete("all")
        canvas.create_image(left, top, image=self.photos[key], anchor="nw")
        if canvas == self.original:
            self.transforms[canvas] = (left, top, dw / width, dh / height, width, height)
            if self.roi:
                x0, y0, x1, y1 = self.roi
                canvas.create_rectangle(left + x0 * dw / width, top + y0 * dh / height,
                                        left + x1 * dw / width, top + y1 * dh / height,
                                        outline="#ffd34e", width=2)
            canvas.create_text(10, 10, anchor="nw", fill="white", text="已暂停：框选取色" if self.frozen else "实时预览")

    def _render(self, frame):
        self.display_frame = frame
        self._draw(self.original, frame, "original")
        hsv = hsv_image(frame, self.blur.get())
        mask = threshold_mask(hsv, self.ranges[self.color_id])
        display_mask = threshold_mask(hsv, self.ranges[self.color_id], clean=True) if self.clean.get() else mask
        result = cv2.bitwise_and(frame, frame, mask=display_mask)
        self._draw(self.mask_canvas, display_mask, "mask")
        self._draw(self.result_canvas, result, "result")
        if not self.valid:
            self.cover_status.set("参数无效：预览暂时沿用上一次有效阈值，请修正后再导出。")
        elif self.roi:
            x0, y0, x1, y1 = self.roi
            selected = mask[y0:y1, x0:x1]
            if selected.size:
                passed = np.count_nonzero(mask)
                inside = np.count_nonzero(selected)
                outside_size = mask.size - selected.size
                outside = (passed - inside) / outside_size * 100 if outside_size else 0
                self.cover_status.set(f"原始阈值通过率：框内 {inside / selected.size * 100:.1f}% · 框外 {outside:.1f}%  | 框外可能有同色物料，需人工检查。")
        else:
            self.cover_status.set(f"{COLORS[self.color_id]} · 全图通过 {np.count_nonzero(mask) / mask.size * 100:.1f}% · {'已开去噪' if self.clean.get() else '原始掩膜'}")

    def _tick(self):
        if self.closed:
            return
        if self.worker:
            frame, status, failed = self.worker.snapshot()
            if self.worker.stop_event.is_set():
                if not self.worker.thread.is_alive():
                    self.worker = None
                    self.camera_status.set(self.image_label or "摄像头已关闭（如有画面，为最后一帧静态图）。")
                elif self.image_label is None:
                    self.camera_status.set("正在等待摄像头驱动释放…")
            else:
                self.camera_status.set(status)
                if frame is not None and not failed:
                    self.raw_frame = frame
                elif failed and not self.frozen and self.raw_frame is not None:
                    self.frozen = True
                    self.frozen_frame = self.raw_frame.copy()
                    self.pause_button.configure(text="恢复实时 / 继续验证")
        frame = self.frozen_frame if self.frozen else self.raw_frame
        if frame is not None:
            self._render(frame)
        self.after_id = self.root.after(50, self._tick)

    def _require_valid(self):
        if not self.valid:
            messagebox.showwarning("参数无效", "先修正HSV上下限，再复制或导出。", parent=self.root)
            return False
        return True

    def _clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.notice.set("已复制。粘贴到main5对应的COLOR_RANGES位置；本软件没有自动修改任何比赛文件。")

    def copy_current(self):
        if self._require_valid():
            self._clipboard(range_entry(self.color_id, self.ranges[self.color_id]))

    def copy_all(self):
        if self._require_valid():
            self._clipboard(export_dictionary(self.ranges))

    def export_text(self):
        if not self._require_valid():
            return
        filename = "现场HSV阈值_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        path = filedialog.asksaveasfilename(parent=self.root, title="导出 main5 阈值", initialfile=filename,
                                          defaultextension=".txt", filetypes=[("文本", "*.txt")])
        if path:
            header = "# OpenCV uint8 HSV：H=0..179，S/V=0..255\n# 仅替换main5的COLOR_RANGES；不修改YOLO。\n"
            header += f"# 取色5x5模糊：{self.blur.get()}；显示掩膜去噪：{self.clean.get()}\n"
            header += "# 未采样颜色可能仍是默认值，现场请逐色验证。\n"
            self._write(path, header + export_dictionary(self.ranges))

    def _write(self, path, contents):
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(contents)
            self.notice.set("已保存：" + path)
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.root)

    def save_config(self):
        if not self._require_valid():
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="保存可重新载入的HSV配置",
                initialfile="现场HSV配置.json", defaultextension=".json", filetypes=[("配置", "*.json")])
        if path:
            data = {"format": "opencv-hsv-main5-v1", "ranges": self.ranges,
                    "blur": self.blur.get(), "clean": self.clean.get()}
            self._write(path, json.dumps(data, ensure_ascii=False, indent=2))

    def load_config(self):
        path = filedialog.askopenfilename(parent=self.root, title="加载本工具保存的配置", filetypes=[("配置", "*.json")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            ranges = config_from_json(data)
            self.ranges = ranges
            self.blur.set(bool(data.get("blur", True)))
            self.clean.set(bool(data.get("clean", True)))
            for groups in self.samples.values():
                groups.clear()
            self._set_values(self.ranges[self.color_id])
            self._update_samples()
            self.notice.set("已载入六种颜色阈值。历史像素样本不在配置内，请现场重新验证。")
        except Exception as exc:
            messagebox.showerror("配置无效，未载入", str(exc), parent=self.root)

    def close(self):
        self.closed = True
        if self.worker:
            self.worker.stop()
        self.root.after_cancel(self.after_id)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="USB摄像头HSV阈值工具")
    parser.add_argument("--demo", action="store_true", help="启动时显示内置演示图，不打开摄像头")
    parser.add_argument("--self-test", metavar="REPORT", help="隐藏窗口的打包版自检，写JSON报告，不打开摄像头")
    args = parser.parse_args()
    root = tk.Tk()
    if args.self_test:
        root.withdraw()
        app = None
        result = {}
        try:
            app = HSVTool(root, demo=True)
            root.update_idletasks()
            app._render(demo_frame())
            app.color_name.set("4  绿(green)")
            app.change_color()
            app.roi = (280, 310, 350, 365)
            app.samples[4].append(roi_samples(hsv_image(demo_frame()), app.roi))
            app.recommend()
            app._render(demo_frame())
            assert app.valid and app.worker is None
            ranges = config_from_json(json.loads(json.dumps({"format": "opencv-hsv-main5-v1", "ranges": app.ranges})))
            assert ranges == app.ranges
            app.values[2].set(255)
            app.values[3].set(100)
            assert not app.valid
            app.restore_default()
            assert app.valid
            app.copy_all()
            assert "COLOR_RANGES" in root.clipboard_get()
            result = {"ok": True, "camera_opened": False, "opencv": cv2.__version__,
                      "tests": ["Tk窗口创建", "三联预览", "颜色切换", "ROI采样推荐", "HSV参数验证", "配置往返", "复制阈值"]}
        except Exception as exc:
            result = {"ok": False, "error": repr(exc)}
        finally:
            if app is not None:
                app.close()
            else:
                root.destroy()
        with open(args.self_test, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        raise SystemExit(0 if result["ok"] else 1)
    HSVTool(root, demo=args.demo)
    root.mainloop()


if __name__ == "__main__":
    main()
