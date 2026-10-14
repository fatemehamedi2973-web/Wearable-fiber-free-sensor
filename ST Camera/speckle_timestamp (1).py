"""VD56G3 live per-frame spatial speckle contrast on Raspberry Pi 5.
Run: python3 speckle_timestamp.py
Requires the same Picamera2/NumPy/Tk environment as temptest, plus Matplotlib
and v4l2-ctl. Enter exposure, gain, FPS and ROI x/y/width/height (raw pixels,
zero-based), or click Select ROI and drag a rectangle on a raw snapshot.
Default ROI is a centered 256x256 box, clipped to the mode size.
Start collecting; Stop drains the bounded ROI-processing queue.

One CSV row per accepted frame: original sensor timestamp in ns, elapsed sensor
time in seconds, frame interval, ROI mean, sample variance (ddof=1), K and K2,
actual controls and quality flags. SensorTimestamp is not wall-clock/UTC.
K = standard deviation / mean; K2 = variance / mean**2 over ROI pixels.
Each run first acquires 100 laser-off frames at the acquisition settings.
Subtract the pixel-wise mean dark map and measured dark/read spatial noise
variance (including 1/100 dark-map estimation uncertainty) from each live frame.
Shot noise and PRNU remain uncorrected. No clipping, smoothing or live averaging.
Raw and corrected relative flow PROXIES (%) = 100 * (1/K2) / mean_baseline(1/K2).
The baseline uses valid unsaturated frames in the first 5 sensor-time seconds,
then stays fixed. No relative values are emitted until baseline completion.
These are inverse-contrast proxies, NOT fitted/calibrated blood flow.
Keep exposure, ROI and illumination fixed. Start a new run for a new baseline.

No live images saved. Dark mean/variance maps are saved for reproducibility.
Only the raw ROI is copied before releasing camera buffers.
Results are buffered and flushed about once per second and on normal Stop;
a crash/power loss can lose recent buffered output. No guaranteed disk fsync.
A bounded 128-ROI/64 MiB queue stops visibly on overload; timestamp gaps are
flagged and never silently filled. Live plotting is throttled and keeps only
10 seconds of history; the CSV retains every processed frame.
Manual sensor mode and sensor digital gain 1x are reapplied and checked.
The original requested exposure/gain/frame-duration tolerances are retained.
Validate measured FPS and gaps on your hardware before a long recording.
"""

import base64
import csv
from collections import deque
import json
import logging
import math
import queue
import re
import shutil
import struct
import subprocess
import threading
import time
import tkinter as tk
import zlib
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from picamera2 import Picamera2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


class SensorManualControl:
    """VD56G3 sensor-side controls, distinct from ISP DigitalGain metadata."""
    def __init__(self):
        self.command = shutil.which("v4l2-ctl")
        if self.command is None:
            raise RuntimeError("v4l2-ctl is required. Install once: sudo apt install v4l-utils")
        devices = []
        for name_file in Path("/sys/class/video4linux").glob("v4l-subdev*/name"):
            name = name_file.read_text(encoding="utf-8").strip()
            if name.split() and name.split()[0].lower() == "vd56g3":
                devices.append((str(Path("/dev") / name_file.parent.name), name))
        if len(devices) != 1:
            raise RuntimeError(
                f"Expected exactly one VD56G3 sensor, found {len(devices)}. "
                "Manual setup cannot safely select a sensor; connect only the intended VD56G3 camera."
            )
        self.device, self.name = devices[0]

    def _call(self, option):
        try:
            result = subprocess.run(
                [self.command, "-d", self.device, option],
                capture_output=True, text=True, timeout=3, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Sensor control failed on {self.device}: {error}") from error
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Sensor control failed on {self.device}: {detail}")
        return result.stdout

    def verify(self):
        output = self._call("--get-ctrl=auto_exposure,digital_gain")
        values = {key: int(value) for key, value in re.findall(
            r"^\s*(auto_exposure|digital_gain)\s*:\s*(-?\d+)\b", output, re.MULTILINE)}
        if values != {"auto_exposure": 1, "digital_gain": 256}:
            raise RuntimeError(
                f"Sensor manual settings did not hold: {values}. "
                "Required auto_exposure=1 (manual), digital_gain=256 (1x). "
                "Check that no other program is controlling the camera."
            )
        return {"device": self.device, "name": self.name, **values,
                "sensor_digital_gain_multiplier": 1.0}

    def apply(self):
        # Separate calls: manual mode must activate the gain control first.
        self._call("--set-ctrl=auto_exposure=1")
        self._call("--set-ctrl=digital_gain=256")
        return self.verify()


def active_raw(raw, stream):
    """Decode unpacked raw and exclude padding for numerical diagnostics."""
    width, height = stream["size"]
    fmt = str(stream["format"])
    if "CSI2P" in fmt or "COMP" in fmt:
        raise ValueError(f"Unsupported packed/compressed raw format: {fmt}")
    if raw.ndim != 2 or raw.shape[0] < height:
        raise ValueError(f"Unexpected raw shape {raw.shape}")
    if raw.dtype == np.uint8:
        if fmt.endswith("8"):
            pixels = raw[:height, :width]
        else:
            if raw.shape[1] < 2 * width:
                raise ValueError("Raw buffer is narrower than the active image")
            pixels = np.ascontiguousarray(raw[:height, :2 * width]).view("<u2")
    elif raw.dtype == np.uint16:
        pixels = raw[:height, :width]
    else:
        raise ValueError(f"Unsupported raw dtype {raw.dtype}")
    if pixels.shape != (height, width):
        raise ValueError(f"Decoded shape {pixels.shape} differs from {(height, width)}")
    return pixels


def make_controls(camera, mode, exposure_ms, gain, fps):
    if not all(math.isfinite(v) and v > 0 for v in (exposure_ms, gain, fps)):
        raise ValueError("Exposure, gain, and FPS must be positive finite numbers.")
    exposure_us, frame_us = math.ceil(exposure_ms * 1000), math.ceil(1_000_000 / fps)
    if exposure_us > frame_us:
        raise ValueError("Exposure exceeds the frame period. Reduce FPS or exposure.")
    if fps > float(mode["fps"]) * 1.001:
        raise ValueError(f"This mode supports approximately {mode['fps']:.3f} FPS maximum.")
    limits = camera.camera_controls
    for key in ("AeEnable", "ExposureTime", "AnalogueGain", "FrameDurationLimits"):
        if key not in limits:
            raise ValueError(f"Camera does not expose required control {key}.")
    for key, value in (("ExposureTime", exposure_us), ("AnalogueGain", gain), ("FrameDurationLimits", frame_us)):
        low, high, _ = limits[key]
        if (low is not None and value < low) or (high is not None and high > 0 and value > high):
            raise ValueError(f"{key}={value} is outside [{low}, {high}] (times in us).")
    result = {"AeEnable": False, "ExposureTime": exposure_us,
              "AnalogueGain": gain, "FrameDurationLimits": (frame_us, frame_us)}
    if "AwbEnable" in limits:
        result["AwbEnable"] = False
    return result


def controls_match(actual, requested):
    for key, target, absolute, relative in (
        ("ExposureTime", requested["ExposureTime"], 100, 0.02),
        ("AnalogueGain", requested["AnalogueGain"], 0.03, 0.05),
        ("FrameDuration", requested["FrameDurationLimits"][0], 200, 0.05),
    ):
        value = actual.get(key)
        if value is None or not math.isfinite(float(value)):
            return False
        if abs(float(value) - target) > max(absolute, relative * target):
            return False
    return True


def validate_roi(roi, size):
    x, y, width, height = roi
    image_width, image_height = size
    if min(x, y) < 0 or min(width, height) < 1 or width * height < 2:
        raise ValueError("ROI needs at least two pixels and nonnegative x/y.")
    if x + width > image_width or y + height > image_height:
        raise ValueError(f"ROI exceeds the raw image size {image_width} x {image_height}.")
    return tuple(roi)


def copy_raw_roi(raw, stream, roi):
    """Decode only ROI rows; own pixels before the camera request is released."""
    x, y, width, height = roi
    row_stream = {**stream, "size": (stream["size"][0], height)}
    pixels = active_raw(raw[y:y + height], row_stream)
    return np.array(pixels[:, x:x + width], copy=True)


def contrast_statistics(roi, saturation_level):
    # Float64 avoids integer overflow; no clipping or dark/noise correction.
    pixels = np.asarray(roi, dtype=np.float64)
    mean = float(pixels.mean())
    variance = float(pixels.var(ddof=1))
    valid = math.isfinite(mean) and mean > 0 and math.isfinite(variance)
    k2 = variance / (mean * mean) if valid else float("nan")
    return {"mean_raw": mean, "variance_raw_ddof1": variance,
            "K": math.sqrt(k2) if valid else float("nan"), "K2": k2,
            "minimum_raw": int(roi.min()), "maximum_raw": int(roi.max()),
            "saturated_fraction": float(np.count_nonzero(roi >= saturation_level) / roi.size),
            "valid_mean": int(valid)}


class DarkCalibration:
    """100 frames at acquisition settings; temporal noise includes dark/read noise.

    Spatially center temporal residuals to include spatial noise correlations.
    No photon shot-noise or multiplicative (PRNU) correction is performed.
    """
    target = 100

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None
        self.spatial_m2 = 0.0
        self.last_timestamp = None

    def add(self, pixels, timestamp, saturation):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Dark sensor timestamps are not strictly increasing")
        if np.any(pixels >= saturation):
            raise ValueError("Dark frame is saturated. Turn off the laser and reduce ambient light.")
        self.last_timestamp = timestamp
        values = np.asarray(pixels, dtype=np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(values)
            self.m2 = np.zeros_like(values)
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        residual = values - self.mean
        self.m2 += delta * residual
        self.spatial_m2 += float(np.sum((delta - delta.mean()) *
                                        (residual - residual.mean()))) / (values.size - 1)

    def save(self, path):
        if self.count != self.target:
            raise ValueError("Dark calibration requires exactly 100 frames")
        variance = self.m2 / (self.count - 1)
        self.noise_variance = self.spatial_m2 / (self.count - 1)
        np.savez_compressed(path / "dark_calibration.npz", mean_dark_raw=self.mean,
                            temporal_variance_raw=variance, frame_count=self.count)
        write_json(path / "dark_calibration.json", {
            "frames": self.count, "mean_dark_raw": float(self.mean.mean()),
            "mean_temporal_variance_raw": float(variance.mean()),
            "spatial_dark_noise_variance_raw": self.noise_variance,
            "dark_read_noise_rms_raw": math.sqrt(float(variance.mean())),
            "note": "Combined dark/read/background noise at acquisition settings, not isolated read noise.",
            "correction": "var(frame - dark_mean, ddof=1) - dark_spatial_noise_variance * (1 + 1/100)",
            "shot_noise_corrected": False, "prnu_corrected": False})

    def correct(self, pixels, saturated_fraction):
        signal = np.asarray(pixels, dtype=np.float64) - self.mean
        mean = float(signal.mean())
        variance = float(signal.var(ddof=1))
        noise = self.noise_variance * (1 + 1 / self.count)
        corrected = variance - noise
        valid = bool(mean > 0 and corrected > 0 and saturated_fraction == 0
                     and math.isfinite(mean) and math.isfinite(corrected))
        k2 = corrected / mean**2 if valid else float("nan")
        return {"mean_dark_subtracted": mean, "variance_dark_subtracted_ddof1": variance,
                "dark_noise_variance_term": noise, "variance_corrected": corrected,
                "K2_corrected": k2, "K_corrected": math.sqrt(k2) if valid else float("nan"),
                "valid_correction": int(valid), "shot_noise_corrected": 0}


class RelativeFlow:
    """Fixed baseline; no rolling normalization or filtering of the signal."""
    def __init__(self, seconds=5.0):
        self.seconds = seconds
        self.total = 0.0
        self.count = 0
        self.baseline = None
        self.state = "collecting"

    def update(self, elapsed, stats):
        k2 = stats["K2"]
        valid = bool(stats["valid_mean"] and math.isfinite(k2) and k2 > 0
                     and stats["saturated_fraction"] == 0)
        inverse = 1.0 / k2 if valid else float("nan")
        valid = valid and math.isfinite(inverse)
        if self.state == "collecting":
            if elapsed < self.seconds:
                if valid:
                    self.count += 1
                    self.total += (inverse - self.total) / self.count  # running mean
            else:
                if self.count >= 2 and self.total > 0 and math.isfinite(self.total):
                    self.baseline = self.total
                    self.state = "ready"
                else:
                    self.state = "failed"
        relative = 100.0 * inverse / self.baseline if valid and self.baseline is not None else float("nan")
        return {"inverse_K2_proxy": inverse, "relative_flow_proxy_percent": relative,
                "baseline_inverse_K2": self.baseline, "baseline_state": self.state,
                "valid_flow_proxy": int(valid)}


class RawSaver:
    """Background ROI analysis and buffered CSV writing; no image files."""
    FIELDS = ["frame_index", "sensor_timestamp_ns", "time_s", "interval_ms",
              "host_received_monotonic_ns", "ExposureTime", "AnalogueGain",
              "DigitalGain", "FrameDuration", "mean_raw", "variance_raw_ddof1",
              "K", "K2", "minimum_raw", "maximum_raw", "saturated_fraction",
              "valid_mean", "timestamp_gap_flag", "inverse_K2_proxy",
              "relative_flow_proxy_percent", "baseline_inverse_K2", "baseline_state", "valid_flow_proxy",
              "mean_dark_subtracted", "variance_dark_subtracted_ddof1", "dark_noise_variance_term",
              "variance_corrected", "K2_corrected", "K_corrected", "valid_correction",
              "shot_noise_corrected", "relative_flow_corrected_percent", "corrected_baseline_state",
              "corrected_inverse_K2_proxy", "corrected_baseline_inverse_K2", "valid_corrected_flow_proxy"]

    def __init__(self, path, stream, roi, bit_depth):
        self.path = path
        frame_bytes = roi[2] * roi[3] * (1 if str(stream["format"]).endswith("8") else 2)
        self.queue = queue.Queue(maxsize=max(1, min(128, (64 * 1024**2) // frame_bytes)))
        self.saturation = 255 if str(stream["format"]).endswith("8") else ((1 << bit_depth) - 1) << (16 - bit_depth)
        self.closing = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.written = self.gaps = 0
        self.mean = self.k2 = self.saturated = None
        self.flow = RelativeFlow()
        self.corrected_flow = RelativeFlow()
        self.dark = None
        self.corrected_k2 = None
        self.corrected_relative = None
        self.relative_flow = None
        self.baseline_status = "collecting"
        self.baseline_value = None
        self.baseline_count = 0
        self.first_ns = self.last_ns = None
        self.trace = deque(maxlen=10000)
        self.thread = threading.Thread(target=self._run, name="speckle-analysis", daemon=False)
        self.thread.start()

    def submit(self, raw, record):
        with self.lock:
            if self.error:
                raise OSError(self.error)
            self.queue.put_nowait((raw, record))

    def snapshot(self, include_trace=False):
        with self.lock:
            fps = (self.written - 1) * 1e9 / (self.last_ns - self.first_ns) if self.written > 1 and self.last_ns > self.first_ns else None
            result = {"written_frames": self.written, "queued_frames": self.queue.qsize(),
                      "last_raw_mean": self.mean, "last_K2": self.k2,
                      "last_K2_corrected": self.corrected_k2,
                      "relative_flow_corrected_percent": self.corrected_relative,
                      "corrected_baseline_state": self.corrected_flow.state,
                      "corrected_baseline_inverse_K2": self.corrected_flow.baseline,
                      "corrected_baseline_valid_frames": self.corrected_flow.count,
                      "relative_flow_proxy_percent": self.relative_flow,
                      "baseline_state": self.baseline_status,
                      "baseline_inverse_K2": self.baseline_value,
                      "baseline_valid_frames": self.baseline_count,
                      "last_saturated_fraction": self.saturated, "timestamp_gap_count": self.gaps,
                      "saved_frame_fps_from_sensor_timestamps": fps, "write_error": self.error}
            if include_trace:
                result["trace"] = list(self.trace)
            return result

    def finish(self):
        self.closing.set()
        self.thread.join()
        return self.snapshot(include_trace=True)

    def _run(self):
        try:
            with (self.path / "speckle.csv").open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=self.FIELDS)
                writer.writeheader()
                last_flush = time.monotonic()
                while not self.closing.is_set() or not self.queue.empty():
                    try:
                        roi, metadata = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        if time.monotonic() - last_flush >= 1:
                            output.flush()
                            last_flush = time.monotonic()
                        continue
                    try:
                        timestamp = int(metadata["SensorTimestamp"])
                        if self.last_ns is not None and timestamp <= self.last_ns:
                            raise ValueError("Sensor timestamps are not strictly increasing")
                        first = timestamp if self.first_ns is None else self.first_ns
                        elapsed = (timestamp - first) / 1e9
                        interval_ms = (timestamp - self.last_ns) / 1e6 if self.last_ns is not None else None
                        expected = metadata.get("FrameDuration")
                        gap = bool(interval_ms is not None and expected and interval_ms * 1000 > 1.5 * expected)
                        stats = contrast_statistics(roi, self.saturation)
                        correction = self.dark.correct(roi, stats["saturated_fraction"])
                        corrected_flow = self.corrected_flow.update(elapsed, {
                            "K2": correction["K2_corrected"],
                            "valid_mean": correction["valid_correction"],
                            "saturated_fraction": stats["saturated_fraction"]})
                        if self.corrected_flow.state != "collecting":
                            if not (self.path / "corrected_baseline.json").exists():
                                write_json(self.path / "corrected_baseline.json", {
                                    "state": self.corrected_flow.state,
                                    "baseline_inverse_K2": self.corrected_flow.baseline,
                                    "valid_frames": self.corrected_flow.count,
                                    "window_seconds": self.corrected_flow.seconds,
                                    "note": "Dark/read-corrected inverse-contrast proxy; shot noise remains."})
                        previous_state = self.flow.state
                        flow_stats = self.flow.update(elapsed, stats)
                        if self.flow.state != previous_state:
                            write_json(self.path / "baseline.json", {
                                "window_seconds": self.flow.seconds, "state": self.flow.state,
                                "valid_frames": self.flow.count, "mean_inverse_K2": self.flow.baseline,
                                "formula": "100 * (1/K2) / mean_baseline(1/K2)",
                                "note": "Uncorrected relative flow proxy; no fitted/calibrated flow."})
                        record = {key: metadata.get(key) for key in self.FIELDS if key in metadata}
                        record.update(stats)
                        record.update(flow_stats)
                        record.update(correction)
                        record.update(relative_flow_corrected_percent=corrected_flow["relative_flow_proxy_percent"],
                                      corrected_baseline_state=self.corrected_flow.state,
                                      corrected_inverse_K2_proxy=corrected_flow["inverse_K2_proxy"],
                                      corrected_baseline_inverse_K2=self.corrected_flow.baseline,
                                      valid_corrected_flow_proxy=corrected_flow["valid_flow_proxy"])
                        record.update(sensor_timestamp_ns=timestamp, time_s=elapsed,
                                      interval_ms=interval_ms, timestamp_gap_flag=int(gap))
                        writer.writerow(record)
                        if time.monotonic() - last_flush >= 1:
                            output.flush()
                            last_flush = time.monotonic()
                        with self.lock:
                            self.written += 1
                            self.mean, self.k2 = stats["mean_raw"], stats["K2"]
                            self.corrected_k2 = correction["K2_corrected"]
                            self.corrected_relative = corrected_flow["relative_flow_proxy_percent"]
                            self.relative_flow = flow_stats["relative_flow_proxy_percent"]
                            self.baseline_status = self.flow.state
                            self.baseline_value, self.baseline_count = self.flow.baseline, self.flow.count
                            self.saturated = stats["saturated_fraction"]
                            self.gaps += int(gap)
                            self.first_ns, self.last_ns = first, timestamp
                            if gap:
                                self.trace.append((elapsed, float("nan"), float("nan"), float("nan"), float("nan")))
                            self.trace.append((elapsed, stats["K2"], self.relative_flow,
                                               self.corrected_k2, self.corrected_relative))
                            while self.trace and self.trace[0][0] < elapsed - 10:
                                self.trace.popleft()
                    finally:
                        self.queue.task_done()
        except Exception as failure:
            with self.lock:
                self.error = f"Speckle processing/saving failed: {failure}"
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break


class CameraWorker(threading.Thread):
    """Sole owner of Picamera2; never calls Tk from a background thread."""
    def __init__(self, messages, shutdown, stop_capture):
        super().__init__(name="camera-worker", daemon=False)
        self.messages, self.shutdown, self.stop_capture = messages, shutdown, stop_capture
        self.commands = queue.Queue()
        self.done = threading.Event()
        self.latest_lock = threading.Lock()
        self.latest = None
        self.camera = None
        self.sensor_manual = None
        self.started = False
        self.modes = []
        self.mode_index = None
        self.run_info = None
        self.last_display = 0.0
        self.logger = logging.getLogger(f"capture.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.log_handler = None

    def _configure(self, index):
        if self.started:
            self.camera.stop()
            self.started = False
        mode = self.modes[index]
        config = self.camera.create_preview_configuration(
            main={"size": (640, 480), "format": "YUV420"},
            raw={"size": mode["size"], "format": mode["unpacked"]},
            sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]},
            buffer_count=4, queue=False,
        )
        self.camera.align_configuration(config)
        self.camera.configure(config)
        self.camera.set_controls({"AeEnable": False})
        self.camera.start()
        self.started = True
        self.sensor_manual.apply()
        self.mode_index = index
        self.messages.put(("ready", index))

    def _begin(self, settings):
        mode = self.modes[self.mode_index]
        controls = make_controls(self.camera, mode, settings["exposure_ms"], settings["gain"], settings["fps"])
        stream = dict(self.camera.stream_configuration("raw"))
        if "CSI2P" in str(stream["format"]) or "COMP" in str(stream["format"]):
            raise ValueError("Camera selected packed/compressed raw. Unpacked raw is required.")
        roi = validate_roi(settings["roi"], stream["size"])
        sensor_settings = self.sensor_manual.apply()
        self.camera.set_controls(controls)
        path = Path(settings["output"]).expanduser() / datetime.now().strftime("speckle_%Y%m%d_%H%M%S_%f")
        path.mkdir(parents=True)
        write_json(path / "metadata.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sensor_mode": {**mode, "index": self.mode_index},
            "requested_controls": controls, "requested_fps": settings["fps"],
            "camera_properties": self.camera.camera_properties,
            "raw_stream": stream,
            "sensor_manual_control": sensor_settings,
            "sensor_manual_check_interval_seconds": 1.0,
            "roi_xywh": roi, "results_file": "speckle.csv", "live_images_saved": False,
            "dark_calibration_file": "dark_calibration.npz",
            "statistics_note": "Raw and dark/read-corrected statistics per frame; sample variance ddof=1. Shot noise and PRNU uncorrected. No smoothing or live averaging.",
            "corrected_formula": "S = frame - pixelwise_mean_dark; K2_corrected = (var(S, ddof=1) - dark_spatial_noise_variance*(1+1/100)) / mean(S)**2",
            "correction_validity": "Positive mean and corrected variance, finite values, no saturated pixels. Invalid corrected K/K2 are NaN.",
            "timestamp_note": "SensorTimestamp nanoseconds; time_s relative to first analyzed frame, not UTC.",
            "formula": "K2 = sample variance / mean**2; K = sqrt(K2)",
            "relative_flow_proxy": {"formula": "100 * (1/K2) / mean_baseline(1/K2)",
                                    "baseline_seconds": 5.0, "baseline": "fixed first 5 seconds; valid unsaturated frames only",
                                    "note": "Uncorrected proxy, not fitted or calibrated blood flow. Keep settings/illumination fixed."},
            "control_verification": {"consecutive_matching_frames": 3, "exposure_tolerance": "max(100 us, 2%)",
                                     "gain_tolerance": "max(0.03x, 5%)", "frame_duration_tolerance": "max(200 us, 5%)"},
        })
        self.log_handler = logging.FileHandler(path / "acquisition.log", encoding="utf-8")
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(self.log_handler)
        self.run_info = {"path": path, "controls": controls, "saver": RawSaver(path, stream, roi, mode["bit_depth"]),
                         "phase": "dark", "dark": DarkCalibration(),
                         "roi": roi, "stream": stream,
                         "accepted": 0, "rejected": 0, "matched": 0, "verified": False,
                         "start": time.monotonic(), "verification_start": time.monotonic(),
                         "first_actual": None, "last_actual": None,
                         "sensor_settings": sensor_settings, "last_sensor_check": 0.0}
        self.logger.info("Requested controls: %s; raw stream: %s", controls, stream)
        self.logger.info("Verified sensor-side manual controls: %s", sensor_settings)
        self.messages.put(("phase", "Checking settings, then collecting 100 laser-off frames..."))

    def _finish(self, reason, error=None):
        info = self.run_info
        if info is None:
            return
        self.messages.put(("phase", "Finishing queued saves..."))
        snapshot = info["saver"].finish()
        error = error or snapshot["write_error"]
        summary = {**{k: v for k, v in snapshot.items() if k != "trace"}, "accepted_frames": info["accepted"],
                   "unsaved_accepted_frames": info["accepted"] - snapshot["written_frames"],
                   "rejected_frames": info["rejected"], "stop_reason": reason, "error": error,
                   "controls_verified": info["verified"], "first_saved_frame_settings": info["first_actual"],
                   "last_observed_settings": info["last_actual"],
                   "last_verified_sensor_manual_control": info["sensor_settings"],
                   "duration_including_setup_and_saving_seconds": time.monotonic() - info["start"],
                   "timestamp_gap_note": "Intervals >1.5 times FrameDuration are flagged; not an exact dropped-frame count."}
        try:
            write_json(info["path"] / "summary.json", summary)
        except Exception as failure:
            error = f"{error + '; ' if error else ''}Could not write summary: {failure}"
        self.logger.info("Run summary: %s", summary)
        self._close_log()
        self.run_info = None
        self.messages.put(("finished", (str(info["path"]), snapshot, error)))

    def _close_log(self):
        if self.log_handler is not None:
            self.logger.removeHandler(self.log_handler)
            self.log_handler.close()
            self.log_handler = None

    def _frame(self):
        request = self.camera.capture_request()
        raw = preview = None
        info = self.run_info
        display = time.monotonic() - self.last_display >= 0.2
        host_ns = time.monotonic_ns()
        try:
            actual = dict(request.get_metadata())
            if display:
                preview = np.array(request.make_array("main"), copy=True)
            if (info is not None and info["verified"] and info["phase"] != "await_laser"
                    and not self.stop_capture.is_set()):
                raw = copy_raw_roi(request.make_array("raw"), info["stream"], info["roi"])
        finally:
            request.release()
        if info is not None and not self.stop_capture.is_set():
            # Check after releasing the request; never save a frame after a failed check.
            now = time.monotonic()
            if now - info["last_sensor_check"] >= 1.0 or not info["verified"] and info["matched"] >= 2:
                try:
                    info["sensor_settings"] = self.sensor_manual.verify()
                    info["last_sensor_check"] = now
                except RuntimeError as failure:
                    info["rejected"] += 1
                    self._finish("Sensor manual control verification failed", str(failure))
                    return
            info["last_actual"] = {k: actual.get(k) for k in ("ExposureTime", "AnalogueGain", "FrameDuration", "DigitalGain")}
            matched = controls_match(actual, info["controls"])
            if not info["verified"]:
                info["matched"] = info["matched"] + 1 if matched else 0
                if info["matched"] >= 3:
                    info["verified"] = True
                    self.logger.info("Verified settings: %s", info["last_actual"])
                    self.messages.put(("phase", "Collecting laser-off calibration" if info["phase"] == "dark" else "Collecting: dark/read corrected; shot noise remains"))
                elif time.monotonic() - info["verification_start"] > max(10.0, 10 * info["controls"]["FrameDurationLimits"][0] / 1e6):
                    self._finish("Settings verification failed", f"Settings did not match. Requested: {info['controls']}. Actual: {info['last_actual']}")
            elif not matched:
                info["rejected"] += 1
                self._finish("Camera settings changed", f"Unexpected camera settings: {info['last_actual']}")
            elif raw is not None:
                if info["phase"] == "dark":
                    try:
                        dark = info["dark"]
                        dark.add(raw, int(actual["SensorTimestamp"]), info["saver"].saturation)
                        self.messages.put(("phase", f"Laser OFF: dark calibration {dark.count}/100 frames"))
                        if dark.count == dark.target:
                            dark.save(info["path"])
                            info["saver"].dark = dark
                            info["phase"] = "await_laser"
                            self.messages.put(("dark_ready", None))
                    except Exception as failure:
                        self._finish("Dark calibration failed", str(failure))
                    return
                if info["phase"] == "await_laser":
                    return
                record = {**actual, "frame_index": info["accepted"], "host_received_monotonic_ns": host_ns}
                try:
                    info["saver"].submit(raw, record)
                    info["accepted"] += 1
                    if info["first_actual"] is None:
                        info["first_actual"] = info["last_actual"].copy()
                except (queue.Full, OSError) as failure:
                    info["rejected"] += 1
                    detail = "ROI processing cannot keep up: queue filled. Reduce ROI size or FPS." if isinstance(failure, queue.Full) else str(failure)
                    self._finish("Saving interrupted", detail)
        if display:
            snapshot = self.run_info["saver"].snapshot(include_trace=True) if self.run_info else None
            with self.latest_lock:
                self.latest = (preview, actual, snapshot)
            self.last_display = time.monotonic()

    def run(self):
        try:
            self.camera = Picamera2()
            if "vd56g3" not in str(self.camera.camera_properties.get("Model", "")).lower():
                raise RuntimeError("This manual-control version requires the active camera to be VD56G3.")
            self.sensor_manual = SensorManualControl()
            self.modes = self.camera.sensor_modes
            if not self.modes:
                raise RuntimeError("No sensor modes available")
            self.messages.put(("modes", self.modes))
            self._configure(0)
            while not self.shutdown.is_set():
                if self.stop_capture.is_set() and self.run_info is not None:
                    self._finish("Stopped by user")
                if self.run_info is not None and self.run_info["saver"].snapshot()["write_error"]:
                    self._finish("Disk write failed")
                try:
                    action, payload = self.commands.get_nowait()
                except queue.Empty:
                    action = None
                if action == "mode" and self.run_info is None:
                    self._configure(payload)
                elif action == "laser_on" and self.run_info is not None:
                    if self.run_info["phase"] == "await_laser" and not self.stop_capture.is_set():
                        self.run_info.update(phase="measurement", verified=False, matched=0,
                                             verification_start=time.monotonic())
                        self.messages.put(("phase", "Laser ON confirmed; checking settings before measurement..."))
                elif action == "roi" and self.run_info is None:
                    request = self.camera.capture_request()
                    try:
                        image = np.array(active_raw(request.make_array("raw"),
                                                    self.camera.stream_configuration("raw")), copy=True)
                    finally:
                        request.release()
                    self.messages.put(("roi_image", image))
                elif action == "start" and self.run_info is None:
                    if self.stop_capture.is_set() or self.shutdown.is_set():
                        self.messages.put(("cancelled", None))
                    else:
                        try:
                            self._begin(payload)
                        except Exception as failure:
                            self._close_log()
                            self.messages.put(("start_error", str(failure)))
                self._frame()
        except Exception as failure:
            self._finish("Camera error", str(failure))
            self.messages.put(("fatal", str(failure)))
        finally:
            try:
                self._finish("Window closed")
            finally:
                self._close_log()
                try:
                    if self.camera is not None:
                        try:
                            if self.started:
                                self.camera.stop()
                        finally:
                            self.camera.close()
                except Exception as failure:
                    self.messages.put(("fatal", f"Camera cleanup: {failure}"))
                finally:
                    self.done.set()


class CameraCaptureUI:
    def __init__(self, root):
        self.root = root
        root.title("Live speckle contrast — raw and dark/read corrected")
        root.minsize(760, 700)
        self.output_path = tk.StringVar(value=str(Path.cwd() / "speckle_results"))
        self.exposure_ms = tk.StringVar(value="5")
        self.analogue_gain = tk.StringVar(value="1")
        self.frame_rate = tk.StringVar(value="30")
        self.roi_vars = [tk.StringVar(value=v) for v in ("0", "0", "128", "128")]
        self.plot_window = None
        self.plot_canvas = None
        self.last_plot = 0.0
        self.status = tk.StringVar(value="Starting camera...")
        self.actual_settings = tk.StringVar(value="Waiting for camera metadata...")
        self.metrics = tk.StringVar(value="Raw intensity and saved-frame FPS appear during collection.")
        self.preview_photo = None
        self.modes = []
        self.busy = self.closing = self.fatal = False
        self.messages = queue.Queue()
        self.shutdown, self.stop_capture = threading.Event(), threading.Event()
        self._build_ui()
        self._open_plot()
        self.worker = CameraWorker(self.messages, self.shutdown, self.stop_capture)
        self.worker.start()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(100, self._refresh_ui)

    @staticmethod
    def _mode_label(mode, index):
        width, height = mode["size"]
        return f"{index}: {width}x{height}, max {mode['fps']:.2f} fps, {mode['bit_depth']}-bit"

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        controls = ttk.Frame(self.root, padding=(12, 10))
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Camera mode").grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.mode_box = ttk.Combobox(controls, state="disabled")
        self.mode_box.grid(row=0, column=1, sticky="ew")
        self.mode_box.bind("<<ComboboxSelected>>", self._mode_changed)
        ttk.Label(controls, text="Save to").grid(row=1, column=0, padx=(0, 8), pady=(10, 0), sticky="w")
        self.path_entry = ttk.Entry(controls, textvariable=self.output_path)
        self.path_entry.grid(row=1, column=1, pady=(10, 0), sticky="ew")
        self.browse_button = ttk.Button(controls, text="Browse...", command=self._browse)
        self.browse_button.grid(row=1, column=2, padx=(8, 0), pady=(10, 0))
        self.entries = [self.path_entry]
        for row, label, variable in ((2, "Exposure time (ms)", self.exposure_ms),
                                     (3, "Analogue gain (x)", self.analogue_gain),
                                     (4, "Frame rate (FPS)", self.frame_rate)):
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=(10, 0), padx=(0, 8))
            entry = ttk.Entry(controls, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", pady=(10, 0))
            self.entries.append(entry)
        roi_box = ttk.Frame(controls)
        roi_box.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Label(roi_box, text="Raw ROI pixels:").pack(side="left")
        for label, variable in zip(("x", "y", "width", "height"), self.roi_vars):
            ttk.Label(roi_box, text=label).pack(side="left", padx=(8, 2))
            entry = ttk.Entry(roi_box, textvariable=variable, width=6)
            entry.pack(side="left")
            self.entries.append(entry)
        ttk.Button(roi_box, text="Show K² plot", command=self._open_plot).pack(side="left", padx=8)
        self.roi_button = ttk.Button(roi_box, text="Select ROI", command=self._select_roi, state="disabled")
        self.roi_button.pack(side="left")
        ttk.Label(controls, text="Settings apply and are verified when you click Start collecting.").grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(controls, textvariable=self.actual_settings, wraplength=690).grid(row=7, column=0, columnspan=3, sticky="w")
        ttk.Label(controls, textvariable=self.metrics, wraplength=690).grid(row=8, column=0, columnspan=3, sticky="w")
        preview_frame = ttk.LabelFrame(self.root, text="Live preview", padding=8)
        preview_frame.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_frame, anchor="center", text="Waiting for camera...")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status, wraplength=480).grid(row=0, column=0, sticky="w")
        self.start_button = ttk.Button(footer, text="Start collecting", command=self._start_capture, state="disabled")
        self.start_button.grid(row=0, column=1, padx=(8, 0))
        self.laser_on_button = ttk.Button(footer, text="Laser is on — start measurement",
                                          command=self._laser_on, state="disabled")
        self.laser_on_button.grid(row=1, column=0, columnspan=3, pady=(8, 0))
        self.stop_button = ttk.Button(footer, text="Stop", command=self._stop_capture, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=(8, 0))

    def _inputs(self, enabled):
        state = "normal" if enabled and not self.closing and not self.fatal else "disabled"
        for entry in self.entries:
            entry.configure(state=state)
        self.browse_button.configure(state=state)
        self.roi_button.configure(state=state)
        self.mode_box.configure(state="readonly" if state == "normal" else "disabled")
        self.start_button.configure(state=state)

    def _mode_changed(self, _event=None):
        self._inputs(False)
        self.status.set("Applying camera mode...")
        self.worker.commands.put(("mode", self.mode_box.current()))

    def _browse(self):
        selected = filedialog.askdirectory(initialdir=self.output_path.get() or str(Path.cwd()))
        if selected:
            self.output_path.set(selected)

    def _select_roi(self):
        self._inputs(False)
        self.status.set("Taking a raw snapshot for ROI selection...")
        self.worker.commands.put(("roi", None))

    def _roi_dialog(self, image):
        from matplotlib.widgets import RectangleSelector
        window = tk.Toplevel(self.root)
        window.title("Drag an ROI on raw image, then Apply")
        window.transient(self.root)
        window.grab_set()
        figure = Figure(figsize=(6, 6), dpi=100)
        axis = figure.add_subplot(111)
        low, high = np.percentile(image, (1, 99))
        axis.imshow(image, cmap="gray", vmin=low, vmax=max(high, low + 1), origin="upper")
        axis.set(xlabel="Raw column (x)", ylabel="Raw row (y)", title="Display contrast normalized; calculations use raw pixels")
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        selection = [tuple(int(v.get()) for v in self.roi_vars)]

        def choose(start, end):
            if None in (start.xdata, start.ydata, end.xdata, end.ydata):
                return
            x0 = max(0, math.floor(min(start.xdata, end.xdata)))
            y0 = max(0, math.floor(min(start.ydata, end.ydata)))
            x1 = min(image.shape[1], math.ceil(max(start.xdata, end.xdata)))
            y1 = min(image.shape[0], math.ceil(max(start.ydata, end.ydata)))
            selection[0] = (x0, y0, x1-x0, y1-y0)

        selector = RectangleSelector(axis, choose, useblit=True, button=[1],
                                     minspanx=1, minspany=1, spancoords="data", interactive=True)
        x, y, w, h = selection[0]
        selector.extents = (x, x+w, y, y+h)
        window.selector = selector  # Keep callback owner alive.

        def apply():
            try:
                validate_roi(selection[0], (image.shape[1], image.shape[0]))
            except ValueError as error:
                messagebox.showerror("Invalid ROI", str(error), parent=window)
                return
            for variable, value in zip(self.roi_vars, selection[0]):
                variable.set(str(value))
            window.destroy()

        ttk.Button(window, text="Apply ROI", command=apply).pack(pady=6)
        canvas.draw()

    def _start_capture(self):
        try:
            settings = {"exposure_ms": float(self.exposure_ms.get()), "gain": float(self.analogue_gain.get()),
                        "fps": float(self.frame_rate.get()), "output": self.output_path.get(),
                        "roi": tuple(int(v.get()) for v in self.roi_vars)}
            validate_roi(settings["roi"], self.modes[self.mode_box.current()]["size"])
            if not settings["output"].strip():
                raise ValueError("Select an output folder.")
        except ValueError as failure:
            messagebox.showerror("Invalid camera settings", str(failure), parent=self.root)
            return
        if not messagebox.askokcancel("Dark calibration — laser OFF",
                "Turn the laser OFF and keep ambient light stable.\n\n"
                "Click OK to collect 100 dark frames at these settings.\n"
                "Keep the laser OFF until calibration finishes.", parent=self.root):
            return
        self.busy = True
        self._open_plot()
        self.plot_line.set_data([], [])
        self.flow_line.set_data([], [])
        self.corrected_plot_line.set_data([], [])
        self.corrected_flow_line.set_data([], [])
        self.plot_canvas.draw_idle()
        self.stop_capture.clear()
        self._inputs(False)
        self.stop_button.configure(state="normal")
        self.metrics.set("Preparing 100-frame laser-off calibration...")
        self.status.set("Applying camera settings...")
        self.worker.commands.put(("start", settings))

    def _laser_on(self):
        self.laser_on_button.configure(state="disabled")
        self.status.set("Laser ON confirmed; preparing measurement...")
        self.worker.commands.put(("laser_on", None))

    def _stop_capture(self):
        self.laser_on_button.configure(state="disabled")
        self.stop_capture.set()
        self.stop_button.configure(state="disabled")
        self.status.set("Stopping capture and finishing queued saves...")

    @staticmethod
    def _preview_data(frame):
        height = frame.shape[0] * 2 // 3
        gray = np.ascontiguousarray(frame[:height, :])
        width = gray.shape[1]
        scanlines = b"".join(b"\x00" + gray[row].tobytes() for row in range(height))
        def chunk(kind, data):
            payload = kind + data
            return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))
        return base64.b64encode(png).decode("ascii")

    def _open_plot(self):
        if self.plot_window is not None and self.plot_window.winfo_exists():
            self.plot_window.lift()
            return
        self.plot_window = tk.Toplevel(self.root)
        self.plot_window.title("Raw and dark/read-corrected results — latest 10 seconds")
        figure = Figure(figsize=(9, 6), dpi=100)
        self.plot_axis = figure.add_subplot(211)
        self.plot_axis.set(xlabel="Time from first measured frame (s)", ylabel="K²",
                           title="Raw and dark/read corrected; shot noise remains")
        self.plot_axis.grid(alpha=0.25)
        self.plot_line, = self.plot_axis.plot([], [], linewidth=0.8, label="Raw")
        self.corrected_plot_line, = self.plot_axis.plot([], [], linewidth=0.8, label="Dark/read corrected")
        self.plot_axis.legend()
        self.flow_axis = figure.add_subplot(212, sharex=self.plot_axis)
        self.flow_axis.set(xlabel="Time from first analyzed frame (s)", ylabel="Relative flow proxy (%)",
                           title="Inverse K² proxies; separate baselines = first 5 s")
        self.flow_axis.axhline(100, color="gray", linestyle="--", linewidth=0.8)
        self.flow_axis.grid(alpha=0.25)
        self.flow_line, = self.flow_axis.plot([], [], color="tab:red", linewidth=0.8, label="Raw")
        self.corrected_flow_line, = self.flow_axis.plot([], [], color="tab:green", linewidth=0.8, label="Dark/read corrected")
        self.flow_axis.legend()
        figure.tight_layout()
        self.plot_canvas = FigureCanvasTkAgg(figure, master=self.plot_window)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.plot_canvas.draw()

    def _show_metrics(self, snapshot):
        fps, mean = snapshot["saved_frame_fps_from_sensor_timestamps"], snapshot["last_raw_mean"]
        fps_text = f"{fps:.2f}" if fps is not None else "waiting"
        mean_text = f"{mean:.2f}" if mean is not None else "waiting"
        k2 = snapshot.get("last_K2")
        k2_text = f"{k2:.6g}" if k2 is not None else "waiting"
        corrected = snapshot.get("last_K2_corrected")
        corrected_text = f"{corrected:.6g}" if corrected is not None and math.isfinite(corrected) else "invalid/waiting"
        corrected_flow = snapshot.get("relative_flow_corrected_percent")
        corrected_flow_text = f"{corrected_flow:.1f}%" if corrected_flow is not None and math.isfinite(corrected_flow) else snapshot.get("corrected_baseline_state", "waiting")
        sat = snapshot.get("last_saturated_fraction")
        sat_text = f"{sat:.1%}" if sat is not None else "waiting"
        flow = snapshot.get("relative_flow_proxy_percent")
        baseline = snapshot.get("baseline_state", "collecting")
        flow_text = f"{flow:.1f}%" if flow is not None and math.isfinite(flow) else "invalid"
        if baseline == "collecting":
            flow_text = "collecting 5 s baseline"
        elif baseline == "failed":
            flow_text = "baseline invalid; check signal and restart"
        self.metrics.set(f"CSV frames: {snapshot['written_frames']} | Queued: {snapshot['queued_frames']} | "
                         f"Measured FPS: {fps_text} | ROI mean: {mean_text} | K²: {k2_text} | "
                         f"Corrected K²: {corrected_text} | Corrected flow: {corrected_flow_text} | "
                         f"Saturated: {sat_text} | Gaps: {snapshot['timestamp_gap_count']} | Raw flow proxy: {flow_text}")
        trace = snapshot.get("trace")
        if trace and self.plot_window is not None and self.plot_window.winfo_exists() and time.monotonic() - self.last_plot >= 0.2:
            values = np.asarray(trace)
            self.plot_line.set_data(values[:, 0], values[:, 1])
            self.flow_line.set_data(values[:, 0], values[:, 2])
            self.corrected_plot_line.set_data(values[:, 0], values[:, 3])
            self.corrected_flow_line.set_data(values[:, 0], values[:, 4])
            end = values[-1, 0]
            self.plot_axis.set_xlim(max(0, end - 10), max(10, end))
            self.plot_axis.relim()
            self.plot_axis.autoscale_view(scalex=False, scaley=True)
            self.flow_axis.relim()
            self.flow_axis.autoscale_view(scalex=False, scaley=True)
            self.plot_canvas.draw_idle()
            self.last_plot = time.monotonic()

    def _refresh_ui(self):
        while True:
            try:
                kind, message = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "modes":
                self.modes = message
                self.mode_box.configure(values=[self._mode_label(m, i) for i, m in enumerate(message)])
                self.mode_box.current(0)
            elif kind == "roi_image":
                self._inputs(True)
                self.status.set("Select ROI, then start collecting.")
                self._roi_dialog(message)
            elif kind == "ready":
                self.mode_box.current(message)
                width, height = self.modes[message]["size"]
                rw, rh = min(256, width), min(256, height)
                for variable, value in zip(self.roi_vars, ((width-rw)//2, (height-rh)//2, rw, rh)):
                    variable.set(str(value))
                self.frame_rate.set(f"{min(30.0, float(self.modes[message]['fps'])):.6g}")
                self._inputs(True)
                self.status.set(f"Previewing {self._mode_label(self.modes[message], message)}")
            elif kind == "phase":
                self.status.set(message)
            elif kind == "dark_ready":
                if self.busy and not self.closing and not self.stop_capture.is_set():
                    self.status.set("100 dark frames collected. Turn the laser ON, then click 'Laser is on — start measurement'.")
                    self.laser_on_button.configure(state="normal")
            elif kind in ("finished", "cancelled", "start_error"):
                self.laser_on_button.configure(state="disabled")
                self.busy = False
                self.stop_button.configure(state="disabled")
                self._inputs(True)
                if kind == "finished":
                    path, snapshot, error = message
                    self._show_metrics(snapshot)
                    self.status.set(f"Saved {snapshot['written_frames']} CSV rows to {path}" if not error else f"Capture stopped: {error}")
                    if error and not self.closing:
                        messagebox.showerror("Capture stopped", error, parent=self.root)
                else:
                    self.status.set("Capture cancelled" if kind == "cancelled" else f"Cannot start: {message}")
                    if kind == "start_error" and not self.closing:
                        messagebox.showerror("Cannot start capture", message, parent=self.root)
            elif kind == "fatal":
                self.laser_on_button.configure(state="disabled")
                self.fatal = True
                self._inputs(False)
                self.stop_button.configure(state="disabled")
                self.status.set(f"Camera error: {message}")
                if not self.closing:
                    messagebox.showerror("Camera error", message, parent=self.root)
        with self.worker.latest_lock:
            latest, self.worker.latest = self.worker.latest, None
        if latest:
            preview, actual, snapshot = latest
            if preview is not None:
                self.preview_photo = tk.PhotoImage(data=self._preview_data(preview), format="PNG")
                self.preview_label.configure(image=self.preview_photo, text="")
            self.actual_settings.set(f"Camera reports: exposure {actual.get('ExposureTime', 'unknown')} us | "
                                     f"gain {actual.get('AnalogueGain', 'unknown')}x | frame duration {actual.get('FrameDuration', 'unknown')} us")
            if snapshot and self.busy:
                self._show_metrics(snapshot)
        if self.closing and self.worker.done.is_set():
            self.root.destroy()
        else:
            self.root.after(100, self._refresh_ui)

    def _close(self):
        self.laser_on_button.configure(state="disabled")
        self.closing = True
        self._inputs(False)
        self.stop_button.configure(state="disabled")
        self.status.set("Closing after queued data finishes saving...")
        self.stop_capture.set()
        self.shutdown.set()


def main():
    root = tk.Tk()
    ui = CameraCaptureUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        ui._close()
    finally:
        ui.stop_capture.set()
        ui.shutdown.set()
        ui.worker.join()


if __name__ == "__main__":
    main()
