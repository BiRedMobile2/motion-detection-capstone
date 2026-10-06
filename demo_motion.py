# Demo: run the motion gate over the saved IR_Test frames.
#
#   python demo_motion.py [image_dir] [--out FIGURE_DIR]
#
# Prints the measured numbers for each pair and writes annotated mask-overlay
# figures so you can see what the outline check is reacting to.
#
# NOTE ON INPUT: the IR_Test JPGs are jet-colormapped display images, which is
# what IRCameraCapture.SaveImages writes alongside the raw CSV. The colormap is
# not monotonic, so this demo inverts it to recover an approximate intensity
# before running the checks. On the rig you skip all of that and feed
# GetTemperatureImage() straight in - see load_frames() below.

import argparse
import os
import sys

import cv2
import numpy as np

import motion_detection as md

# Pairs to evaluate: (label, reference file, comparison file, what it shows)
PAIRS = [
    ("still", "IR_Test.jpg", "IR_Test4.jpg", "same angle, subject held still"),
    ("moved", "IR_Test2.jpg", "IR_Test3.jpg", "same angle, subject shifted"),
]

# Overlay colors, BGR.
SHARED = (150, 150, 150)
A_ONLY = (232, 210, 79)
B_ONLY = (60, 132, 245)
BACKDROP = (28, 28, 28)


def _jet_lut():
    ramp = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(ramp, cv2.COLORMAP_JET).reshape(256, 3).astype(np.int32)


def dejet(path):
    """Recover approximate intensity from a jet-colormapped JPG.

    Demo scaffolding only. Real captures come from GetTemperatureImage() as
    floats in degrees C and need none of this.
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w, _ = bgr.shape
    diff = (bgr.astype(np.int32).reshape(-1, 1, 3) - _jet_lut().reshape(1, 256, 3)) ** 2
    return diff.sum(axis=2).argmin(axis=1).astype(np.uint8).reshape(h, w)


def load_frames(image_dir, *names):
    """Load saved frames for the demo.

    On the rig, replace this with either the raw temperature CSVs that
    SaveImages writes:

        np.loadtxt(path, delimiter=",")

    or the live pointer from the SDK:

        A = np.ctypeslib.as_array(ptr, shape=(ht, wd))
    """
    return [dejet(os.path.join(image_dir, n)) for n in names]


def overlay(mask_a, mask_b, label_a="A", label_b="B"):
    """Paint two body masks on top of each other with a legend."""
    h, w = mask_a.shape
    out = np.full((h, w, 3), BACKDROP, np.uint8)
    out[mask_a & mask_b] = SHARED
    out[mask_a & ~mask_b] = A_ONLY
    out[mask_b & ~mask_a] = B_ONLY

    entries = [(SHARED, "both"), (A_ONLY, label_a + " only"), (B_ONLY, label_b + " only")]
    for row, (color, text) in enumerate(entries):
        y = 14 + row * 26
        cv2.rectangle(out, (12, y), (32, y + 18), color, -1)
        cv2.putText(out, text, (40, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return out


def run(image_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    thresholds = md.Thresholds()

    print(f"\nthresholds: IoU >= {thresholds.min_iou}, "
          f"shift <= {thresholds.max_shift_px} px  (provisional, not yet calibrated)\n")
    header = f"{'case':<7} {'IoU':>7} {'shift px':>9} {'dx':>8} {'dy':>8}  verdict"
    print(header)
    print("-" * len(header))

    results = []
    for label, file_a, file_b, blurb in PAIRS:
        frame_a, frame_b = load_frames(image_dir, file_a, file_b)

        # The gate runs on a burst; two saved frames stand in for one here.
        burst = md.check_burst([frame_a, frame_b], thresholds, skin_temp_c=None)
        pair = burst.pairs[0]

        verdict = "PASS - keep frame" if burst.passed else "FAIL - " + burst.reasons[0]
        print(f"{label:<7} {pair.iou:>7.3f} {pair.shift_px:>9.2f} "
              f"{pair.dx:>8.2f} {pair.dy:>8.2f}  {verdict}")

        mask_a = md.body_mask(frame_a, skin_temp_c=None)
        mask_b = md.body_mask(frame_b, skin_temp_c=None)
        path = os.path.join(out_dir, f"overlay_{label}.png")
        cv2.imwrite(path, overlay(mask_a, mask_b,
                                  os.path.splitext(file_a)[0],
                                  os.path.splitext(file_b)[0]))
        results.append((label, blurb, path, burst))

    print("\nwhat each case is:")
    for label, blurb, path, _ in results:
        print(f"  {label:<7} {blurb:<34} -> {path}")

    # Sensitivity: slide one real frame by a known amount and see what comes back.
    print("\nsensitivity - a known slide applied to IR_Test:\n")
    base = load_frames(image_dir, "IR_Test.jpg")[0]
    h, w = base.shape
    print(f"{'true':>8} {'IoU':>8} {'measured':>10} {'error':>8}  verdict")
    print("-" * 48)
    for px in (0, 2, 5, 10, 20, 40):
        matrix = np.float32([[1, 0, px], [0, 1, 0]])
        slid = cv2.warpAffine(base, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        pair = md.compare(base, slid, thresholds=thresholds, skin_temp_c=None)
        print(f"{px:>5} px {pair.iou:>8.3f} {pair.shift_px:>7.2f} px "
              f"{abs(pair.shift_px - px):>5.2f} px  {'pass' if pair.passed else 'FAIL'}")

    # End-of-session closure check, standing in the first frame for both ends.
    closure = md.check_closure(base, base, thresholds, skin_temp_c=None)
    print(f"\nclosure check (same frame both ends, sanity only): "
          f"IoU {closure.iou:.3f}, shift {closure.shift_px:.2f} px, "
          f"{'pass' if closure.passed else 'FAIL'}")
    print(f"\nfigures written to {out_dir}/\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", nargs="?", default=".",
                        help="folder holding the IR_Test JPGs (default: current dir)")
    parser.add_argument("--out", default="motion_figs",
                        help="where to write the overlay figures")
    args = parser.parse_args()

    try:
        run(args.image_dir, args.out)
    except FileNotFoundError as exc:
        print(f"could not find {exc}. Pass the folder holding the IR_Test JPGs:\n"
              f"  python demo_motion.py path/to/images", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
