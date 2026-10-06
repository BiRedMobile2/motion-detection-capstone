# Tests for the capture layer in IRCameraCapture.py, with Qt and the SDK stubbed.
# Run with:  python -m pytest test_capture_stubbed.py
# or plain:  python test_capture_stubbed.py
#
# Covers the queue, GrabBurst, the retry loop, recording and the signals. Does
# not cover the ctypes marshalling, which needs the real DLL.
#
# The stubs below have to be installed BEFORE importing IRCameraCapture, because
# that module loads the DLLs when its class body runs.

import os
import shutil
import sys
import tempfile
import types

import cv2
import numpy as np

# -- stub PySide6 ------------------------------------------------------------


class Signal:
    """Stand-in for Qt's Signal that records what was emitted."""

    def __set_name__(self, owner, name):
        self._name = name

    def __init__(self, *args):
        self._name = None

    def __get__(self, obj, owner):
        if obj is None:
            return self
        return obj.__dict__.setdefault("_sig_" + self._name, _BoundSignal())


class _BoundSignal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)

    def connect(self, handler):
        pass


class QThread:
    def __init__(self):
        pass

    def start(self):
        pass

    def quit(self):
        pass


def _install_stubs():
    qtcore = types.ModuleType("PySide6.QtCore")
    qtcore.QThread = QThread
    qtcore.Signal = Signal
    qtcore.Qt = types.SimpleNamespace(KeepAspectRatio=1)

    qtwidgets = types.ModuleType("PySide6.QtWidgets")
    qtwidgets.QWidget = type("QWidget", (), {})
    qtwidgets.QLabel = type("QLabel", (), {})

    qtgui = types.ModuleType("PySide6.QtGui")
    qtgui.QImage = type("QImage", (), {
        "__init__": lambda self, *a: None,
        "scaled": lambda self, *a: self,
        "Format_BGR888": 1,
    })
    qtgui.QPixmap = type("QPixmap", (), {})

    pyside = types.ModuleType("PySide6")
    for name, module in (("QtCore", qtcore), ("QtWidgets", qtwidgets), ("QtGui", qtgui)):
        setattr(pyside, name, module)
        sys.modules["PySide6." + name] = module
    sys.modules["PySide6"] = pyside

    import ctypes

    class FakeDLL:
        def __getattr__(self, name):
            return lambda *a, **k: 0

    ctypes.cdll.LoadLibrary = lambda path: FakeDLL()


_install_stubs()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import IRCameraCapture as ic  # noqa: E402  (must follow the stubs)


# -- helpers -----------------------------------------------------------------

BURST = dict(count=3, seconds=0.0)   # small and instant, for the tests


def frame(cx=320, temp_body=33.0, temp_room=22.0):
    """A warm blob on a cool background, in degrees C."""
    f = np.full((480, 640), temp_room, np.float32)
    cv2.ellipse(f, (cx, 240), (120, 170), 0, 0, 360, float(temp_body), -1)
    return f


class ScriptedReader(ic.irCameraReader):
    """A reader whose frames come from a list instead of the camera."""

    def __init__(self, script):
        super().__init__()
        self.script = script
        self.reads = 0
        self.ht, self.wd = 480, 640
        self.ThreadActive = True

    def ReadTemperatureFrame(self):
        f = self.script[min(self.reads, len(self.script) - 1)]
        self.reads += 1
        return f


def capture_at(reader, position=4, retries=3):
    """Run one gated capture with the burst shrunk to keep the test quick."""
    reader.GrabBurst = lambda **k: ic.irCameraReader.GrabBurst(reader, **BURST)
    reader._run_position(position, settle=0.0, retries=retries)


def still_reader(n=3):
    s = frame()
    return ScriptedReader([s] + [s.copy() for _ in range(n - 1)])


def moving_reader():
    return ScriptedReader([frame(cx=320), frame(cx=400), frame(cx=250)] * 4)


def tmpdir():
    return tempfile.mkdtemp(prefix="ircap_test_")


# -- the gated capture -------------------------------------------------------


def test_still_burst_passes():
    r = still_reader()
    capture_at(r)
    assert len(r.BurstFinished.calls) == 1
    assert len(r.OperatorAlert.calls) == 0


def test_passing_burst_hands_back_a_keeper_frame():
    r = still_reader()
    capture_at(r)
    position, result, keeper, frames = r.BurstFinished.calls[0]
    assert position == 4
    assert isinstance(keeper, np.ndarray)
    assert result.passed


def test_every_burst_frame_is_handed_back():
    r = still_reader()
    capture_at(r)
    _, _, _, frames = r.BurstFinished.calls[0]
    assert len(frames) == 3
    assert len(r.last_burst) == 3


def test_first_capture_is_stashed_for_closure():
    r = still_reader()
    capture_at(r)
    assert r.first_capture is not None


def test_moving_burst_retries_then_alerts():
    r = moving_reader()
    capture_at(r)
    assert len(r.BurstFinished.calls) == 0
    assert len(r.OperatorAlert.calls) == 1
    assert "attempts failed" in r.OperatorAlert.calls[0][1]


def test_every_attempt_is_logged():
    r = moving_reader()
    capture_at(r)
    assert len(r.log) == 3, "three attempts should produce three log rows"


def test_dead_camera_alerts_instead_of_hanging():
    class DeadReader(ScriptedReader):
        def ReadTemperatureFrame(self):
            return None

    r = DeadReader([frame()])
    r.GrabBurst = lambda **k: []
    r._run_position(1, settle=0.0, retries=3)
    assert len(r.OperatorAlert.calls) == 1


# -- the closure check -------------------------------------------------------


def test_closure_without_a_reference_alerts():
    r = still_reader()
    r._run_closure(settle=0.0)
    assert len(r.OperatorAlert.calls) == 1
    assert len(r.ClosureFinished.calls) == 0


def test_closure_after_a_capture_reports():
    r = still_reader()
    capture_at(r)
    r.GrabBurst = lambda **k: ic.irCameraReader.GrabBurst(r, **BURST)
    r._run_closure(settle=0.0)
    assert len(r.ClosureFinished.calls) == 1


# -- the queue ---------------------------------------------------------------


def test_capture_position_queues_work():
    r = still_reader()
    r.CapturePosition(7)
    with r._lock:
        assert r._pending == ("position", 7)


def test_capture_closure_queues_work():
    r = still_reader()
    r.CaptureClosure()
    with r._lock:
        assert r._pending == ("closure", None)


# -- saving ------------------------------------------------------------------


def test_save_burst_writes_npy_and_jpg_per_frame():
    folder = tmpdir()
    try:
        r = still_reader()
        capture_at(r)
        assert r.SaveBurst(folder, "b") == 3
        written = sorted(os.listdir(folder))
        assert len(written) == 6, written
        assert sum(f.endswith(".npy") for f in written) == 3
        assert sum(f.endswith(".jpg") for f in written) == 3
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_save_log_writes_a_row_per_attempt():
    folder = tmpdir()
    try:
        r = moving_reader()
        capture_at(r)
        path = os.path.join(folder, "log.csv")
        r.SaveLog(path)
        rows = open(path).read().strip().split("\n")
        assert rows[0] == "position,attempt,iou,shift_px,passed"
        assert len(rows) == 4, "header plus three attempts"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# -- continuous recording ----------------------------------------------------


def _record(reader, folder, count, **kwargs):
    reader.StartRecording(folder, "rec", **kwargs)
    with reader._lock:
        rec = reader._recording
    for _ in range(count):
        with reader._lock:
            if reader._recording is None:
                break
        reader._record_frame(rec, reader.ReadTemperatureFrame())
    return rec


def test_recording_writes_every_frame():
    folder = tmpdir()
    try:
        r = still_reader(4)
        _record(r, folder, 4, interval=0.0, save_jpg=True, log_metrics=False)
        assert r.StopRecording() == 4
        assert len(os.listdir(folder)) == 8, "npy + jpg for each of 4 frames"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_recording_logs_metrics_between_consecutive_frames():
    folder = tmpdir()
    try:
        s = frame()
        r = ScriptedReader([s, s.copy(), frame(cx=400), s.copy()])
        _record(r, folder, 4, interval=0.0, save_jpg=False, log_metrics=True)
        r.StopRecording()
        assert len(r.recording_log) == 3, "4 frames make 3 pairs"
        shifts = [row["shift_px"] for row in r.recording_log]
        assert any(s > 5 for s in shifts), f"the moved frame should show up: {shifts}"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_recording_log_writes_csv():
    folder = tmpdir()
    try:
        s = frame()
        r = ScriptedReader([s, s.copy(), frame(cx=400)])
        _record(r, folder, 3, interval=0.0, save_jpg=False, log_metrics=True)
        r.StopRecording()
        path = os.path.join(folder, "metrics.csv")
        r.SaveRecordingLog(path)
        rows = open(path).read().strip().split("\n")
        assert rows[0] == "frame,iou,shift_px,dx,dy,passed"
        assert len(rows) == 3, "header plus two pairs"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_stop_recording_fires_once():
    folder = tmpdir()
    try:
        r = still_reader()
        _record(r, folder, 2, interval=0.0, save_jpg=False, log_metrics=False)
        r.StopRecording()
        assert len(r.RecordingStopped.calls) == 1
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_max_frames_stops_the_recording():
    folder = tmpdir()
    try:
        r = still_reader()
        rec = _record(r, folder, 5, max_frames=2, save_jpg=False, log_metrics=False)
        assert rec["count"] == 2
        with r._lock:
            assert r._recording is None, "recording should have stopped itself"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# -- compatibility with the original API -------------------------------------


def test_old_pausefeed_spelling_still_works():
    r = still_reader()
    r.pauseFeed(True)
    assert r.feedNoShow is True


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  pass  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
