# Grabs a few frames at one physical position, then measures the positions
# against each other.
#
#   python grab.py d00            take 10 frames, label them d00
#   python grab.py d05            move the subject first, then take 10 more
#   python grab.py --curve        compare every position against the first
#   python grab.py --curve --ref r00
#   python grab.py --curve --neighbor    compare each position to the one before
#
# Written for the phantom tests: move the subject by a measured amount, grab,
# repeat. The curve then says what a millimeter or a degree of real movement
# costs in IoU, which is the one thing the volunteer runs could never pin down.
#
# --neighbor exists because a cooling heat source drifts. Measured against one
# reference taken minutes earlier, every later position carries all the drift
# since. Measured against its immediate neighbor, seconds earlier, it carries
# almost none. Interleave the reference position between test positions
# (r00a, r01, r00b, r02, ...) and --neighbor reads the ladder off cleanly.
#
# Positions land in grabs/<label>/. Ten frames at 512x640 is about 13 MB.

import argparse
import ctypes as ct
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_detection as md

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
FRAMES = 10
SETTLE_READS = 10


def open_camera():
    ct.cdll.LoadLibrary(os.path.join(THIS_DIR, "opencv_core440.dll"))
    dll = ct.cdll.LoadLibrary(os.path.join(THIS_DIR, "icisdk_x64.dll"))
    dll.StartCamera(0, 1)
    height, width = dll.GetImgHeight(), dll.GetImgWidth()
    if height <= 0 or width <= 0:
        dll.StopCamera()
        raise SystemExit(f"camera reported a bad size: {height} x {width}")
    dll.GetTemperatureImage.restype = ct.POINTER(ct.c_float)
    return dll, height, width


def make_grabber(dll, height, width):
    def grab():
        pointer = dll.GetTemperatureImage()
        if not pointer:
            return None
        raw = np.ctypeslib.as_array(pointer, shape=(height, width)).copy()
        return cv2.rotate(cv2.flip(raw, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
    return grab


def take(args):
    folder = os.path.join(args.out, args.label)
    if os.path.exists(folder) and glob.glob(os.path.join(folder, "f_*.npy")):
        raise SystemExit(f"'{args.label}' already has frames - pick another label")
    os.makedirs(folder, exist_ok=True)

    dll, height, width = open_camera()
    grab = make_grabber(dll, height, width)
    try:
        for _ in range(SETTLE_READS):      # let the camera settle
            grab()
            time.sleep(0.05)

        kept = 0
        while kept < args.frames:
            frame = grab()
            if frame is None:
                continue
            np.save(os.path.join(folder, f"f_{kept:03d}.npy"), frame)
            kept += 1
            time.sleep(0.05)
    finally:
        # Always hand the camera back, or every later run inherits one that is
        # still open and reports a size of 0 x 0.
        dll.StopCamera()

    first = np.load(os.path.join(folder, "f_000.npy"))
    print(f"{args.label}: {kept} frames, {height} x {width}, "
          f"median {np.median(first):.0f}, hottest {first.max():.0f}")

    jpg = os.path.join(folder, "preview.jpg")
    eight_bit = cv2.normalize(first, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(jpg, cv2.applyColorMap(cv2.cvtColor(eight_bit, cv2.COLOR_GRAY2BGR),
                                       cv2.COLORMAP_JET))
    print(f"  {jpg}   <- open this and check the subject is in frame and warm")
    return 0


def representative(folder):
    """The middle frame of a position, so one noisy grab cannot skew the curve."""
    paths = sorted(glob.glob(os.path.join(folder, "f_*.npy")))
    if not paths:
        return None
    return np.load(paths[len(paths) // 2])


def tripped_by(pair):
    names = []
    if pair.iou < md.MIN_IOU:
        names.append("outline")
    if pair.shift_px > md.MAX_SHIFT_PX:
        names.append("shift")
    return " + ".join(names)


def curve(args):
    folders = sorted(d for d in glob.glob(os.path.join(args.out, "*"))
                     if os.path.isdir(d))
    if len(folders) < 2:
        raise SystemExit(f"need at least two positions in {args.out}")
    labels = [os.path.basename(d) for d in folders]
    frames = {l: representative(d) for l, d in zip(labels, folders)}

    if args.neighbor:
        # Each position against the one before it. Drift between two grabs
        # taken seconds apart is small, so what is left is the real movement.
        print("each position measured against the one before it\n")
        print(f"{'pair':>26}{'IoU':>9}{'shift px':>11}{'verdict':>10}   tripped by")
        for before, after in zip(labels, labels[1:]):
            if frames[before] is None or frames[after] is None:
                continue
            pair = md.compare(frames[before], frames[after])
            verdict = "pass" if pair.passed else "REJECT"
            print(f"{before + ' -> ' + after:>26}{pair.iou:9.3f}"
                  f"{pair.shift_px:11.2f}{verdict:>10}   {tripped_by(pair)}")
        print(f"\nthresholds: IoU >= {md.MIN_IOU}, shift <= {md.MAX_SHIFT_PX} px")
        print("Neighbour mode is for a drifting heat source. Interleave the")
        print("reference position between test positions and read the ladder")
        print("off the pairs that span a real move.")
        return 0

    reference_label = args.ref or labels[0]
    if reference_label not in labels:
        raise SystemExit(f"no position called '{reference_label}' - "
                         f"have: {', '.join(labels)}")
    reference = frames[reference_label]
    print(f"everything measured against '{reference_label}'\n")
    print(f"{'position':>12}{'IoU':>9}{'shift px':>11}{'verdict':>10}   tripped by")

    crossed = None
    for label in labels:
        frame = frames[label]
        if frame is None:
            continue
        pair = md.compare(reference, frame)
        verdict = "pass" if pair.passed else "REJECT"
        print(f"{label:>12}{pair.iou:9.3f}{pair.shift_px:11.2f}"
              f"{verdict:>10}   {tripped_by(pair)}")
        if crossed is None and label != reference_label and not pair.passed:
            crossed = (label, tripped_by(pair))

    print(f"\nthresholds: IoU >= {md.MIN_IOU}, shift <= {md.MAX_SHIFT_PX} px")
    if crossed:
        label, by = crossed
        print(f"first position rejected: {label}, on the {by} check")
        print("That is the gate's sensitivity. Convert the label back to millimeters")
        print("or degrees and you have the number in physical units.")
        print("\nIf the readings get steadily worse down the list and the shift stays")
        print("near zero, that is the heat source drifting, not movement. Re-run with")
        print("--neighbor.")
    else:
        print("nothing was rejected - every position still looks like the reference.")
        print("Either the moves were too small to see, or the subject did not move.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?", help="name for this position, e.g. d05")
    parser.add_argument("--curve", action="store_true",
                        help="measure the positions already grabbed")
    parser.add_argument("--ref", help="position to measure against (default: the first)")
    parser.add_argument("--neighbor", action="store_true",
                        help="compare each position to the one before it")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--out", default=os.path.join(THIS_DIR, "grabs"))
    args = parser.parse_args()

    if args.curve:
        return curve(args)
    if not args.label:
        parser.error("give a label to grab, or --curve to measure what you grabbed")
    return take(args)


if __name__ == "__main__":
    raise SystemExit(main())
