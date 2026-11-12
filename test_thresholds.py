"""不调用真实摄像头的回归测试：python -m unittest -v test_thresholds"""
import ast
import copy
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from thresholds import *


class ThresholdTests(unittest.TestCase):
    def test_default_round_trip(self):
        for ranges in DEFAULT_RANGES.values():
            original = threshold_mask(np.uint8([[[h, 120, 150] for h in range(180)]]), ranges)
            recovered = threshold_mask(np.uint8([[[h, 120, 150] for h in range(180)]]),
                                       controls_to_ranges(ranges_to_controls(ranges)))
            np.testing.assert_array_equal(original, recovered)

    def test_red_wrap(self):
        hues = [178, 179, 0, 1] * 100
        lo, hi = shortest_hue_arc(hues, 100, 2)
        self.assertGreater(lo, hi)
        samples = np.uint8([[h, 200, 180] for h in hues])
        ranges, _ = recommend_ranges(samples, 1, 100, 2, 10, 10)
        self.assertEqual(len(ranges), 2)
        mask = threshold_mask(np.uint8([[[179, 200, 180], [0, 200, 180], [90, 200, 180]]]), ranges)
        self.assertEqual(mask.tolist(), [[255, 255, 0]])

    def test_outlier_hue(self):
        lo, hi = shortest_hue_arc([58] * 995 + [140] * 5, 98, 4)
        self.assertEqual((lo, hi), (54, 62))

    def test_hue_full_circle(self):
        self.assertEqual(shortest_hue_arc(np.arange(180), 100, 4), (0, 179))

    def test_invalid_controls_and_config(self):
        for controls in [(0, 10, 80, 70, 0, 255), (0, 10, 0, 255, 80, 70),
                         (180, 10, 0, 255, 0, 255), (-1, 10, 0, 255, 0, 255)]:
            with self.assertRaises(ValueError):
                controls_to_ranges(controls)
        with self.assertRaises(ValueError):
            validate_ranges([(0, 1)])
        with self.assertRaises(ValueError):
            ranges_to_controls([((0, 0, 0), (100, 255, 255)), ((90, 0, 0), (179, 255, 255))])
        with self.assertRaises(ValueError):
            config_from_json({"format": "wrong"})

    def test_black_range(self):
        ranges, warnings = recommend_ranges(np.uint8([[90, 20, 30]] * 100), 5)
        self.assertEqual(ranges[0][0], (0, 0, 0))
        self.assertEqual(ranges[0][1], (179, 180, 45))
        self.assertTrue(warnings)

    def test_gray_rejected_for_color(self):
        with self.assertRaises(ValueError):
            recommend_ranges(np.uint8([[0, 0, 220]] * 100), 1)

    def test_roi_clamp_and_limit(self):
        hsv = np.zeros((50, 100, 3), np.uint8)
        samples = roi_samples(hsv, (-10, -10, 200, 60), 100)
        self.assertEqual(samples.shape, (100, 3))
        with self.assertRaises(ValueError):
            roi_samples(hsv, (1, 1, 3, 3))

    def test_preprocessing(self):
        frame = np.random.default_rng(1).integers(0, 256, (50, 60, 3), dtype=np.uint8)
        before = frame.copy()
        expected = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        np.testing.assert_array_equal(hsv_image(frame), expected)
        np.testing.assert_array_equal(frame, before)

    def test_exports(self):
        tree = ast.parse(export_dictionary(DEFAULT_RANGES))
        self.assertEqual(ast.literal_eval(tree.body[0].value), DEFAULT_RANGES)
        data = json.loads(json.dumps({"format": "opencv-hsv-main5-v1", "ranges": DEFAULT_RANGES}))
        self.assertEqual(config_from_json(data), DEFAULT_RANGES)


class CameraTests(unittest.TestCase):
    def test_failed_camera_released(self):
        from camera_worker import CameraWorker
        with patch("camera_worker.cv2.VideoCapture") as capture:
            capture.return_value.isOpened.return_value = False
            worker = CameraWorker(1)
            worker._run()
            self.assertTrue(worker.snapshot()[2])
            capture.return_value.release.assert_called_once()

    def test_stop_does_not_block(self):
        from camera_worker import CameraWorker
        worker = CameraWorker(1)
        worker.stop()
        self.assertTrue(worker.stop_event.is_set())


class GUITests(unittest.TestCase):
    def test_demo_and_controls(self):
        import tkinter as tk
        from hsv_tool import HSVTool, demo_frame
        root = tk.Tk()
        root.withdraw()
        app = HSVTool(root, demo=True)
        try:
            root.update_idletasks()
            self.assertIsNone(app.worker)  # 不自动访问真实摄像头
            self.assertEqual(app.display_frame.shape, (480, 640, 3))
            app.color_name.set("4  绿(green)")
            app.change_color()
            app.roi = (280, 310, 350, 365)
            app.samples[4].append(roi_samples(hsv_image(demo_frame()), app.roi))
            app.recommend()
            self.assertTrue(app.valid)
            app._render(demo_frame())
            app.values[2].set(255)
            app.values[3].set(100)
            self.assertFalse(app.valid)
            app.restore_default()
            self.assertTrue(app.valid)
            app.copy_all()
            self.assertIn("COLOR_RANGES", root.clipboard_get())
            app.blur.set(False)
            app.preprocessing_changed()
            self.assertFalse(app.samples[4])
            app._render(demo_frame())
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
