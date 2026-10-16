# Calibration recorder. Captures a labelled run straight from the camera and
# measures every frame against the one before it.
#
#   python calibrate.py record still_01 --seconds 60
#   python calibrate.py record moving_small_01 --seconds 60
#   python calibrate.py compare runs/still_01/f_00000.npy runs/still_01/f_00300.npy
#
# Deliberately does NOT use Qt or IRCameraCapture - it talks to the SDK
# directly, so a calibration session cannot be blocked by a UI problem.
#
# The numbers it writes are what MIN_IOU and MAX_SHIFT_PX should be set from.

import argparse
import ctypes as ct
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_detection as md

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def open_camera():
    ct.cdll.LoadLibrary(os.path.join(THIS_DIR, "opencv_core440.dll"))
    dll = ct.cdll.LoadLibrary(os.path.join(THIS_DIR, "icisdk_x64.dll"))
    dll.StartCamera(0, 1)
    height, width = dll.GetImgHeight(), dll.GetImgWidth()
    if height <= 0 or width <= 0:
        raise RuntimeError(f"camera reported a bad size: {height} x {width}")
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


def record(args):
    folder = os.path.join(args.out, args.name)
    os.makedirs(folder, exist_ok=True)

    dll, height, width = open_camera()
    grab = make_grabber(dll, height, width)

    for _ in range(10):          # let the camera settle before we start
        grab()
        time.sleep(0.05)

    print(f"recording '{args.name}' for {args.seconds}s - go   (ctrl-c stops early)")
    rows, previous, count = [], None, 0
    history = deque()        # (stamp, frame) going back one burst span
    started = time.perf_counter()

    try:
        while time.perf_counter() - started < args.seconds:
            frame = grab()
            if frame is None:
                continue

            stamp = time.perf_counter() - started
            if not args.metrics_only:
                np.save(os.path.join(folder, f"f_{count:05d}.npy"), frame)

            # Two gaps, because two things need calibrating. Consecutive frames
            # are ~37 ms apart and show the noise floor. The span pair matches
            # what check_burst actually does - every frame against frames[0]
            # across a whole burst - and that is what the thresholds govern.
            span_iou = span_shift = span_conf = None
            history.append((stamp, frame))
            while len(history) >= 2 and history[1][0] <= stamp - args.burst_seconds:
                history.popleft()
            if stamp - history[0][0] >= args.burst_seconds * 0.9:
                span = md.compare(history[0][1], frame)
                span_iou, span_shift, span_conf = span.iou, span.shift_px, span.confidence

            if previous is not None:
                pair = md.compare(previous, frame)
                rows.append((count, round(stamp, 3), pair.iou, pair.shift_px,
                             pair.dx, pair.dy, pair.confidence,
                             span_iou, span_shift, span_conf))
                if count % 20 == 0:
                    extra = ("" if span_iou is None else
                             f"   | over {args.burst_seconds:g}s  "
                             f"IoU {span_iou:.3f}  shift {span_shift:6.2f} px")
                    print(f"  {stamp:5.1f}s  frame {count:4d}  "
                          f"IoU {pair.iou:.3f}  shift {pair.shift_px:6.2f} px{extra}")

            previous = frame
            count += 1
    except KeyboardInterrupt:
        print("\n  stopped early - keeping what was recorded")
    finally:
        # Always hand the camera back. A camera left open makes every later
        # run report a size of 0 x 0, and the shutter keeps clicking.
        dll.StopCamera()

    elapsed = time.perf_counter() - started

    csv_path = os.path.join(folder, "metrics.csv")
    with open(csv_path, "w") as handle:
        handle.write("frame,seconds,iou,shift_px,dx,dy,confidence,"
                     "span_iou,span_shift_px,span_confidence\n")
        for row in rows:
            tail = (",,," if row[7] is None else
                    ",{:.4f},{:.3f},{:.4f}".format(row[7], row[8], row[9]))
            handle.write("{},{},{:.4f},{:.3f},{:.3f},{:.3f},{:.4f}".format(*row[:7])
                         + tail + "\n")

    print(f"\n{count} frames in {elapsed:.1f}s ({count / max(elapsed, 0.001):.1f} fps)")
    summarize(rows, args.burst_seconds)
    print(f"\nwrote {csv_path}")
    if not args.metrics_only:
        size_mb = count * height * width * 4 / 1e6
        print(f"frames in {folder}  (~{size_mb:.0f} MB)")
    return 0


def summarize(rows, burst_seconds=1.0):
    if not rows:
        print("no pairs measured")
        return

    def block(label, ious, shifts, confs):
        # The interesting tail differs: IoU is bad when LOW, shift when HIGH,
        # and confidence is bad when LOW.
        print(f"\n{label}")
        print(f"{'':10}{'median':>9}{'worst 5%':>11}{'worst':>9}")
        print(f"{'IoU':10}{np.median(ious):9.3f}{np.percentile(ious, 5):11.3f}"
              f"{ious.min():9.3f}")
        print(f"{'shift px':10}{np.median(shifts):9.2f}"
              f"{np.percentile(shifts, 95):11.2f}{shifts.max():9.2f}")
        print(f"{'conf':10}{np.median(confs):9.3f}{np.percentile(confs, 5):11.3f}"
              f"{confs.min():9.3f}")

    block("frame to frame (noise floor)",
          np.array([r[2] for r in rows]),
          np.array([r[3] for r in rows]),
          np.array([r[6] for r in rows]))

    span = [r for r in rows if r[7] is not None]
    if span:
        block(f"across {burst_seconds:g}s (what check_burst sees)",
              np.array([r[7] for r in span]),
              np.array([r[8] for r in span]),
              np.array([r[9] for r in span]))
        print(f"\nset MIN_IOU below the worst-5% IoU of the {burst_seconds:g}s block")
        print("on a still run, and MAX_SHIFT_PX above its worst-5% shift.")
        print("The frame-to-frame block is the noise floor, not the threshold.")
    else:
        print(f"\nrun was shorter than {burst_seconds:g}s - no span pairs to report.")


def compare(args):
    a = np.load(args.first)
    b = np.load(args.second)
    pair = md.compare(a, b)
    print(f"IoU       {pair.iou:.3f}")
    print(f"shift     {pair.shift_px:.2f} px  (dx {pair.dx:.2f}, dy {pair.dy:.2f})")
    print(f"verdict   {'pass' if pair.passed else 'FAIL'}")
    for reason in pair.reasons:
        print(f"  - {reason}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="record a labelled run")
    rec.add_argument("name", help="run label, e.g. still_01")
    rec.add_argument("--seconds", type=float, default=60.0)
    rec.add_argument("--out", default=os.path.join(THIS_DIR, "runs"))
    rec.add_argument("--burst-seconds", type=float, default=1.0,
                     help="burst window to measure across; match BURST_SECONDS")
    rec.add_argument("--metrics-only", action="store_true",
                     help="skip saving frames; just write metrics.csv")
    rec.set_defaults(func=record)

    cmp_ = sub.add_parser("compare", help="measure two saved .npy frames")
    cmp_.add_argument("first")
    cmp_.add_argument("second")
    cmp_.set_defaults(func=compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
