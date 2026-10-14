"""Headless numerical and acquisition tests; no Picamera2 or Tk required."""
import ast
import csv
import json
import logging
import math
import queue
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path

import numpy as np

source = Path(__file__).with_name("real_time_flow_monitoing_with_NC.py")
tree = ast.parse(source.read_text(encoding="utf-8"))
names = {"write_json", "active_raw", "copy_raw_roi", "contrast_statistics",
         "controls_match", "DarkCalibration", "RelativeFlow", "RawSaver", "CameraWorker"}
exec(compile(ast.Module(body=[node for node in tree.body if
     isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names],
     type_ignores=[]), str(source), "exec"), globals())


class NoiseCorrectionTests(unittest.TestCase):
    def test_dark_statistics_and_correction(self):
        rng = np.random.default_rng(12)
        frames = 50 + np.arange(12).reshape(3, 4) + rng.normal(0, 2, (100, 3, 4))
        dark = DarkCalibration()
        for i, frame in enumerate(frames):
            dark.add(frame, i + 1, 1023)
        with tempfile.TemporaryDirectory() as folder:
            dark.save(Path(folder))
            with np.load(Path(folder) / "dark_calibration.npz") as maps:
                np.testing.assert_allclose(maps["mean_dark_raw"], frames.mean(axis=0))
                np.testing.assert_allclose(maps["temporal_variance_raw"], frames.var(axis=0, ddof=1))
        residuals = frames - frames.mean(axis=0)
        expected_noise = np.sum(np.var(residuals, axis=(1, 2), ddof=1)) / 99
        self.assertAlmostEqual(dark.noise_variance, expected_noise)
        signal = np.arange(12).reshape(3, 4) + 20
        result = dark.correct(dark.mean + signal, 0)
        expected_variance = signal.var(ddof=1) - expected_noise * 1.01
        self.assertAlmostEqual(result["K2_corrected"], expected_variance / signal.mean()**2)
        self.assertTrue(result["valid_correction"])
        for pixels, saturation in [(dark.mean - 1, 0), (dark.mean + 20, 0),
                                    (dark.mean + signal, 0.01)]:
            result = dark.correct(pixels, saturation)
            self.assertFalse(result["valid_correction"])
            self.assertTrue(math.isnan(result["K2_corrected"]))
        self.assertEqual(result["shot_noise_corrected"], 0)

    def test_dark_rejects_saturation_and_duplicate_timestamps(self):
        dark = DarkCalibration()
        with self.assertRaises(ValueError):
            dark.add(np.full((2, 2), 255), 1, 255)
        dark.add(np.zeros((2, 2)), 1, 255)
        with self.assertRaises(ValueError):
            dark.add(np.zeros((2, 2)), 1, 255)

    def test_csv_and_corrected_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            dark = DarkCalibration()
            for i in range(100):
                dark.add(np.full((2, 2), 10), i + 1, 255)
            dark.save(path)
            saver = RawSaver(path, {"format": "Y8"}, (0, 0, 2, 2), 8)
            saver.dark = dark
            for i in range(7):
                saver.submit(np.array([[20, 30], [40, 50]], dtype=np.uint8), {
                    "SensorTimestamp": 1_000_000_000 * (i + 1),
                    "FrameDuration": 1_000_000, "frame_index": i})
            snapshot = saver.finish()
            self.assertIsNone(snapshot["write_error"])
            self.assertEqual(snapshot["written_frames"], 7)
            with (path / "speckle.csv").open() as output:
                rows = list(csv.DictReader(output))
            self.assertEqual(len(rows), 7)
            self.assertAlmostEqual(float(rows[-1]["relative_flow_corrected_percent"]), 100)
            self.assertNotEqual(rows[-1]["K2"], rows[-1]["K2_corrected"])
            self.assertEqual(rows[-1]["mean_dark_subtracted"], "25.0")
            self.assertEqual(len(snapshot["trace"][-1]), 5)

    def test_worker_dark_phase_never_saves_measurement(self):
        class Request:
            def __init__(self, index):
                self.index = index
                self.released = False
            def get_metadata(self):
                return {"SensorTimestamp": self.index, "ExposureTime": 1000,
                        "AnalogueGain": 1, "FrameDuration": 10000}
            def make_array(self, stream):
                return np.full((2, 2), 10, dtype=np.uint8)
            def release(self):
                self.released = True
        class Camera:
            index = 0
            def capture_request(self):
                self.index += 1
                self.last = Request(self.index)
                return self.last
        class Manual:
            def verify(self):
                return {}
        with tempfile.TemporaryDirectory() as folder:
            worker = CameraWorker(queue.Queue(), threading.Event(), threading.Event())
            worker.camera, worker.sensor_manual = Camera(), Manual()
            saver = RawSaver(Path(folder), {"format": "Y8"}, (0, 0, 2, 2), 8)
            worker.run_info = info = {
                "path": Path(folder), "phase": "dark", "dark": DarkCalibration(),
                "saver": saver, "verified": True, "matched": 3,
                "controls": {"ExposureTime": 1000, "AnalogueGain": 1,
                             "FrameDurationLimits": (10000, 10000)},
                "roi": (0, 0, 2, 2), "stream": {"format": "Y8", "size": (2, 2)},
                "last_sensor_check": 0, "accepted": 0, "rejected": 0,
                "first_actual": None}
            try:
                for _ in range(100):
                    worker._frame()
                    self.assertTrue(worker.camera.last.released)
                self.assertEqual(info["phase"], "await_laser")
                worker._frame()
                self.assertEqual(info["dark"].count, 100)
                self.assertEqual(info["accepted"], 0)
                info["phase"] = "measurement"
                worker._frame()
                self.assertEqual(info["accepted"], 1)
            finally:
                snapshot = saver.finish()
            self.assertIsNone(snapshot["write_error"])


if __name__ == "__main__":
    unittest.main()
