# Tests for the motion gate. Run with:  python -m pytest test_motion_detection.py
# or plain:  python test_motion_detection.py

import cv2
import numpy as np

import motion_detection as md


def synthetic_frame(cx=320, cy=240, temp_body=33.0, temp_room=22.0):
    """A warm blob on a cool background, in degrees C."""
    frame = np.full((480, 640), temp_room, np.float32)
    cv2.ellipse(frame, (cx, cy), (120, 170), 0, 0, 360, float(temp_body), -1)
    return frame


def slide(frame, dx, dy=0):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def test_identical_frames_pass():
    frame = synthetic_frame()
    result = md.check_burst([frame, frame.copy(), frame.copy()])
    assert result.passed
    assert result.worst_iou == 1.0
    assert result.worst_shift < 0.01


def test_large_slide_is_rejected():
    frame = synthetic_frame()
    moved = slide(frame, 40)
    assert not md.check_burst([frame, moved, moved.copy()]).passed


def test_two_frame_burst_can_still_fail():
    """Regression: a short burst must not disable rejection.

    consecutive_failures defaults to 2, but a two-frame burst yields only one
    pair. Before the fix the counter never reached 2 and every short burst
    passed no matter how badly it moved.
    """
    frame = synthetic_frame()
    assert not md.check_burst([frame, slide(frame, 60)]).passed


def test_single_frame_is_not_a_burst():
    result = md.check_burst([synthetic_frame()])
    assert not result.passed
    assert "at least 2 frames" in result.reasons[0]


def test_phase_correlation_recovers_a_known_slide():
    frame = synthetic_frame()
    for dx in (2, 5, 10, 25):
        _, _, measured, _ = md.pixel_shift(frame, slide(frame, dx))
        assert abs(measured - dx) < 0.5, f"{dx} px slide read back as {measured:.2f}"


def test_uniform_warming_is_not_motion():
    """Every pixel 3 C warmer must not register as movement."""
    frame = synthetic_frame()
    warmer = frame + 3.0
    result = md.check_burst([frame, warmer, warmer.copy()])
    assert result.passed, "temperature drift was mistaken for motion"


def test_absolute_skin_threshold_matches_otsu_on_clean_frames():
    frame = synthetic_frame()
    absolute = md.body_mask(frame, skin_temp_c=28.0)
    otsu = md.body_mask(frame, skin_temp_c=None)
    assert md.mask_iou(absolute, otsu) > 0.95


def test_transient_bad_frame_does_not_sink_the_burst():
    """One shutter-event frame in the middle should be tolerated."""
    frame = synthetic_frame()
    burst = [frame, frame.copy(), slide(frame, 80), frame.copy(), frame.copy()]
    assert md.check_burst(burst).passed


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
