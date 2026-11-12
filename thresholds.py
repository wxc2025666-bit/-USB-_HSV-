"""OpenCV uint8 HSV阈值计算与main5格式输出，不依赖GUI或摄像头。"""
import math
import numpy as np
import cv2

COLORS = {1: "红(red)", 2: "黄(yellow)", 3: "深蓝(dblue)",
          4: "绿(green)", 5: "黑(black)", 6: "浅蓝(lblue)"}
DEFAULT_RANGES = {
    1: [((0, 80, 50), (10, 255, 255)), ((170, 80, 50), (179, 255, 255))],
    2: [((19, 80, 60), (41, 255, 255))],
    3: [((101, 80, 45), (130, 255, 255))],
    4: [((40, 60, 45), (78, 255, 255))],
    5: [((0, 0, 0), (179, 180, 60))],
    6: [((80, 35, 100), (100, 140, 255))],
}


def validate_ranges(ranges):
    if not isinstance(ranges, (list, tuple)) or not 1 <= len(ranges) <= 2:
        raise ValueError("每种颜色需要1~2个HSV区间")
    result = []
    for item in ranges:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("区间必须为(下限,上限)")
        low, high = item
        if not isinstance(low, (list, tuple)) or not isinstance(high, (list, tuple)) or len(low) != 3 or len(high) != 3:
            raise ValueError("HSV上下限都需要三个整数")
        for i, maximum in enumerate((179, 255, 255)):
            if any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                   for v in (low[i], high[i])):
                raise ValueError("HSV值必须是整数")
            if not 0 <= low[i] <= high[i] <= maximum:
                raise ValueError("HSV区间超限或上下限倒置")
        result.append((tuple(int(v) for v in low), tuple(int(v) for v in high)))
    return result


def controls_to_ranges(values):
    """H下限>H上限时跨越179/0，自动拆成两段；S/V必须顺序正确。"""
    h0, h1, s0, s1, v0, v1 = map(int, values)
    if s0 > s1 or v0 > v1:
        raise ValueError("S和V的下限不能大于上限；H倒置表示跨越179/0")
    hue_parts = [(h0, h1)] if h0 <= h1 else [(h0, 179), (0, h1)]
    return validate_ranges([((lo, s0, v0), (hi, s1, v1)) for lo, hi in hue_parts])


def ranges_to_controls(ranges):
    ranges = validate_ranges(ranges)
    if len(ranges) == 1:
        lo, hi = ranges[0]
        return lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]
    parts = sorted(ranges, key=lambda r: r[0][0])
    first, second = parts
    if first[0][0] != 0 or second[1][0] != 179 or first[0][1:] != second[0][1:] or first[1][1:] != second[1][1:] or first[1][0] >= second[0][0]:
        raise ValueError("两段范围应跨越179/0，且S/V限制一致，才能用六个滑块表示")
    return second[0][0], first[1][0], first[0][1], first[1][1], first[0][2], first[1][2]


def shortest_hue_arc(hues, coverage=98.0, margin=4):
    """求覆盖指定比例样本的最短色相弧，不把红色179/0平均成青色。"""
    data = np.sort(np.asarray(hues, dtype=np.int32).reshape(-1))
    if not len(data) or np.any((data < 0) | (data > 179)):
        raise ValueError("没有有效色相样本")
    if not 50 <= float(coverage) <= 100 or int(margin) < 0:
        raise ValueError("覆盖率应为50~100%，余量不能为负")
    count = max(1, int(math.ceil(len(data) * float(coverage) / 100.0)))
    extended = np.concatenate((data, data + 180))
    starts = np.arange(len(data))
    widths = extended[starts + count - 1] - extended[starts]
    start = int(np.argmin(widths))
    low = int(extended[start]) - int(margin)
    high = int(extended[start + count - 1]) + int(margin)
    if high - low >= 179:
        return 0, 179
    return low % 180, high % 180


def recommend_ranges(samples, color_id, coverage=98.0, h_pad=4, s_pad=15, v_pad=15):
    pixels = np.asarray(samples, dtype=np.uint8).reshape(-1, 3)
    if len(pixels) < 16:
        raise ValueError("样本太少，请在物料顶面拖一个稍大的框")
    if color_id not in COLORS:
        raise ValueError("未知颜色编号")
    if not 50 <= float(coverage) <= 100 or min(h_pad, s_pad, v_pad) < 0:
        raise ValueError("覆盖率或余量无效")
    warnings = []
    if color_id != 5:
        # 灰白高光和过暗像素的H不可靠，不拿它们推算有色物料的色相。
        valid = (pixels[:, 1] >= 30) & (pixels[:, 2] >= 25)
        fraction = float(valid.mean())
        if fraction < 0.65:
            warnings.append("选框内低饱和/过暗像素较多，请避开反光、阴影和背景")
        pixels = pixels[valid]
        if len(pixels) < 16:
            raise ValueError("有效有色样本太少，请重新框选物料顶面，或检查选中的颜色")
        h0, h1 = shortest_hue_arc(pixels[:, 0], coverage, h_pad)
    else:
        h0, h1 = 0, 179
        warnings.append("黑色主要看V上限；HSV不能区分黑物料、阴影与麦轮，仍需main5的ROI/形状筛选")
    trim = (100.0 - float(coverage)) / 2.0
    low = np.floor(np.percentile(pixels[:, 1:], trim, axis=0)).astype(int)
    high = np.ceil(np.percentile(pixels[:, 1:], 100.0 - trim, axis=0)).astype(int)
    s0, s1 = max(0, int(low[0]) - int(s_pad)), min(255, int(high[0]) + int(s_pad))
    v0, v1 = max(0, int(low[1]) - int(v_pad)), min(255, int(high[1]) + int(v_pad))
    if color_id == 5:
        s0 = v0 = 0
        # 暗像素S容易波动，保留main5原有180的宽松S上限，再允许样本要求更大。
        s1 = max(180, s1)
    elif color_id in (3, 6):
        warnings.append("深蓝/浅蓝可能重叠，生成后请同时验证另一种蓝色不会通过")
    return controls_to_ranges((h0, h1, s0, s1, v0, v1)), warnings


def hsv_image(frame, blur=True):
    work = cv2.GaussianBlur(frame, (5, 5), 0) if blur else frame
    return cv2.cvtColor(work, cv2.COLOR_BGR2HSV)


def threshold_mask(hsv, ranges, clean=False):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in validate_ranges(ranges):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))
    if clean:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return mask


def roi_samples(hsv, roi, limit=8000):
    height, width = hsv.shape[:2]
    x0, y0, x1, y1 = map(int, roi)
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("请拖出至少4×4像素的取色框")
    pixels = hsv[y0:y1, x0:x1].reshape(-1, 3)
    if len(pixels) > limit:
        pixels = pixels[np.linspace(0, len(pixels) - 1, limit, dtype=int)]
    return pixels.copy()


def range_entry(color_id, ranges):
    return "{}: {},  # {}".format(color_id, repr(validate_ranges(ranges)), COLORS[color_id])


def export_dictionary(all_ranges):
    return "COLOR_RANGES = {\n" + "\n".join("    " + range_entry(i, all_ranges[i]) for i in COLORS) + "\n}\n"


def config_from_json(data):
    if not isinstance(data, dict) or data.get("format") != "opencv-hsv-main5-v1":
        raise ValueError("不是本工具导出的HSV配置")
    result = {}
    for color_id in COLORS:
        result[color_id] = validate_ranges(data["ranges"][str(color_id)])
        ranges_to_controls(result[color_id])
    return result
