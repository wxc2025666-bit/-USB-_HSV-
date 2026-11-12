"""摄像头仅在后台线程读取，GUI不会积压旧帧；不开串口。"""
import threading
import time
import cv2


class CameraWorker:
    def __init__(self, index, backend="DSHOW", size=(640, 480)):
        self.index, self.backend, self.size = int(index), backend, size
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.status = "正在打开摄像头…"
        self.failed = False
        self.control_request = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def snapshot(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy(), self.status, self.failed)

    def set_exposure(self, auto, exposure):
        with self.lock:
            self.control_request = (bool(auto), float(exposure))

    def _status(self, message, failed=False):
        with self.lock:
            self.status, self.failed = message, failed

    def _run(self):
        cap = None
        try:
            api = cv2.CAP_DSHOW if self.backend == "DSHOW" else cv2.CAP_MSMF
            cap = cv2.VideoCapture(self.index, api)
            if not cap.isOpened():
                self._status("打不开设备{}；请试其他编号/后端，并关闭占用摄像头的软件".format(self.index), True)
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            cap.set(cv2.CAP_PROP_FPS, 30)
            failures = 0
            while not self.stop_event.is_set():
                with self.lock:
                    control = self.control_request
                    self.control_request = None
                if control is not None:
                    auto, exposure = control
                    accepted = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto else 0.25)
                    if not auto:
                        accepted = cap.set(cv2.CAP_PROP_EXPOSURE, exposure) and accepted
                    self._status("曝光设置已提交（是否生效依摄像头驱动而定）" if accepted else "驱动未接受曝光设置；可用摄像头厂商软件调整")
                ok, frame = cap.read()
                if not ok or frame is None or not frame.size:
                    failures += 1
                    if failures >= 30:
                        self._status("摄像头连续读取失败；检查连接后重新打开", True)
                        return
                    time.sleep(0.03)
                    continue
                failures = 0
                with self.lock:
                    first = self.frame is None
                    self.frame = frame
                    if first:
                        self.status = "已连接设备{}，实际{}×{}；取色时保持现场曝光和高度".format(self.index, frame.shape[1], frame.shape[0])
        except Exception as exc:
            self._status("摄像头异常：{}".format(exc), True)
        finally:
            if cap is not None:
                cap.release()
