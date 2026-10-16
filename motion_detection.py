# Measures whether the patient moved between two IR frames.
#
# Hand it two frames from the same camera angle and it reports two numbers:
#
#   outline overlap  how much the two body outlines agree, 0 to 1
#   pixel shift      how far the image slid, in pixels
#
# Neither number is thrown off by the skin slowly warming or cooling, which is
# what makes this harder than it first looks.
#
# Only ever compare frames from the SAME camera angle. Across angles, the rig's
# own rotation reads as patient movement.
#
# No camera, no Qt, no SDK - just numpy and OpenCV, so it runs anywhere and can
# be tested without hardware.

from dataclasses import dataclass, field

import cv2
import numpy as np

# Calibrated 2026-10-08 against the camera, five 60-second runs with a
# volunteer. Figures are the worst case over a 1-second burst window:
#
#              IoU median   IoU worst   shift worst
#   still            0.996       0.987       0.77 px
#   breathing        0.994       0.981       1.16 px
#   small moves      0.993       0.735    (unusable, see below)
#   large moves      0.992       0.372    (unusable, see below)
#
# IoU does the real work. Still and breathing never fell below 0.981 across
# 5000+ pairs; movement reached 0.372. MIN_IOU sits in the gap.
MIN_IOU = 0.97

# Secondary. Phase correlation measures how far the WHOLE SCENE translated, and
# most of the scene is stationary room and rig, so a patient turning in place
# barely registers - large moves scored IoU 0.42 while reporting half a pixel.
# When it does break down it returns nonsense (1622 px on a 640x512 image).
# Small moves did produce honest 2-3 px translations, which is what this cut is
# for; it is not a substitute for the IoU check.
MAX_SHIFT_PX = 2.0

# Tells "didn't move" apart from "couldn't tell". A blank or blown-out frame
# gives phaseCorrelate nothing to lock onto, and the shift it returns is often
# near zero - which would otherwise look like a perfectly still patient and
# pass. Below this confidence, the shift is rejected instead of believed.
#
# Measured across four runs. The response is NOT bounded to 0-1:
#
#   still        never below 1.03
#   breathing    worst 0.42
#   movement     worst -0.04, with 5-16 pairs per run under this cut
#
# A negative response means no correlation peak was found at all. 0.01 sits in
# the dead zone between breathing and outright failure, so the guard fires only
# when the measurement has collapsed - never on a patient who merely moved.
MIN_CONFIDENCE = 0.01

# Skin runs ~30-35 C against a room at ~20-24 C, so 28 sits between them.
#
# Unused by default. The camera on the bench reports raw counts (~3000-3900),
# not degrees, so body_mask falls back to Otsu. Pass skin_temp_c=SKIN_TEMP_C
# explicitly once the frames really are Celsius.
SKIN_TEMP_C = 28.0


@dataclass
class Thresholds:
    """Limits a burst must satisfy to be kept."""

    min_iou: float = MIN_IOU
    max_shift_px: float = MAX_SHIFT_PX
    min_confidence: float = MIN_CONFIDENCE
    # How many consecutive failing pairs before the burst is rejected. Breathing
    # moves the chest wall slightly and periodically; requiring two in a row
    # stops one noisy frame from throwing the capture away.
    consecutive_failures: int = 2


@dataclass
class PairResult:
    """One frame compared against the burst's reference frame."""

    index: int
    iou: float
    dx: float
    dy: float
    shift_px: float
    confidence: float
    passed: bool
    reasons: list = field(default_factory=list)


@dataclass
class BurstResult:
    """Verdict for a whole burst captured at one camera position."""

    passed: bool
    pairs: list = field(default_factory=list)
    keeper: int = 0  # index of the frame to keep when passed
    reasons: list = field(default_factory=list)

    @property
    def worst_iou(self) -> float:
        return min((p.iou for p in self.pairs), default=1.0)

    @property
    def worst_shift(self) -> float:
        return max((p.shift_px for p in self.pairs), default=0.0)

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}  IoU {self.worst_iou:.3f} (min {MIN_IOU})  "
            f"shift {self.worst_shift:.2f} px (max {MAX_SHIFT_PX})"
        )


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Per-frame min-max stretch to 8-bit.

    Stretching each frame on its own is deliberate: it removes the slow warming
    and cooling of the skin over a session, so drift cannot look like motion.
    """
    if frame.dtype == np.uint8:
        return frame
    return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def body_mask(frame: np.ndarray, skin_temp_c: float = None) -> np.ndarray:
    """Boolean mask of the warm body against the cooler room.

    frame        temperature array in degrees C, or an 8-bit display image
    skin_temp_c  absolute cut in degrees C. Pass None (or hand in an 8-bit
                 frame) to let Otsu pick the cut from the histogram instead.
    """
    if skin_temp_c is not None and frame.dtype != np.uint8:
        mask = (frame >= skin_temp_c).astype(np.uint8) * 255
    else:
        blurred = cv2.GaussianBlur(to_uint8(frame), (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Drop speckle without eating the real outline.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two body masks. 1.0 means identical."""
    union = int((a | b).sum())
    if union == 0:
        return 0.0
    return float((a & b).sum()) / union


def pixel_shift(a: np.ndarray, b: np.ndarray):
    """How far frame b slid relative to frame a.

    Returns (dx, dy, magnitude, confidence), all in pixels except confidence.
    Phase correlation works on the frequency-domain pattern rather than raw
    values, so warming and cooling do not register as movement. It measures
    translation only - not rotation, not a change of viewpoint.
    """
    fa = to_uint8(a).astype(np.float32)
    fb = to_uint8(b).astype(np.float32)
    window = cv2.createHanningWindow((fa.shape[1], fa.shape[0]), cv2.CV_32F)
    (dx, dy), confidence = cv2.phaseCorrelate(fa, fb, window)
    return dx, dy, float(np.hypot(dx, dy)), float(confidence)


def compare(reference, frame, index=1, thresholds=None, skin_temp_c=None):
    """Run both checks on one pair of frames from the same camera position."""
    thresholds = thresholds or Thresholds()

    iou = mask_iou(body_mask(reference, skin_temp_c), body_mask(frame, skin_temp_c))
    dx, dy, shift, confidence = pixel_shift(reference, frame)

    reasons = []
    if iou < thresholds.min_iou:
        reasons.append(f"outline moved (IoU {iou:.3f} < {thresholds.min_iou})")
    if shift > thresholds.max_shift_px:
        reasons.append(f"image slid {shift:.2f} px (max {thresholds.max_shift_px})")
    if confidence < thresholds.min_confidence:
        reasons.append(f"shift unreliable (confidence {confidence:.3f})")

    return PairResult(index, iou, dx, dy, shift, confidence, not reasons, reasons)


def check_burst(frames, thresholds=None, skin_temp_c=None) -> BurstResult:
    """Decide whether a burst from one camera position is usable.

    frames  list of 2+ frames captured back to back at ONE position, oldest
            first. Temperature arrays in degrees C, or 8-bit display images.

    Every frame is compared against the first. The burst is rejected once
    `consecutive_failures` pairs fail in a row, which keeps one noisy frame
    (or a shutter event) from costing a whole capture.
    """
    thresholds = thresholds or Thresholds()

    if len(frames) < 2:
        return BurstResult(False, [], 0, ["need at least 2 frames to detect motion"])

    reference = frames[0]
    pairs = [
        compare(reference, frame, i, thresholds, skin_temp_c)
        for i, frame in enumerate(frames[1:], start=1)
    ]

    # Never ask for more consecutive failures than there are pairs to fail: a
    # two-frame burst yields one pair, and requiring two in a row there would
    # silently disable rejection altogether.
    required = max(1, min(thresholds.consecutive_failures, len(pairs)))

    run = 0
    for pair in pairs:
        run = run + 1 if not pair.passed else 0
        if run >= required:
            failing = [p for p in pairs if not p.passed]
            return BurstResult(False, pairs, 0, failing[-1].reasons)

    # Clean burst. The frames are aligned, so the middle one is as good as any
    # and sits furthest from any settling wobble at the edges of the window.
    return BurstResult(True, pairs, len(frames) // 2, [])


def check_closure(first_capture, closing_capture, thresholds=None,
                  skin_temp_c=None) -> PairResult:
    """End-of-session check: did the patient shift BETWEEN positions?

    The per-position checks never compare one angle to another, so they cannot
    see someone resettling between stops. Returning the rig to its first angle
    and re-shooting gives a matching viewpoint to compare against the very first
    capture of the session.

    This assumes the rig returns to the same position repeatably. Measure that
    with an empty rig before trusting the result.
    """
    return compare(first_capture, closing_capture, -1, thresholds, skin_temp_c)
