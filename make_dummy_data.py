"""Make dummy motion sequences from the real IR shots, then score loftr_motion on them.

Each case is a known flow field (dx, dy per pixel). The expected verdict comes
from that flow with the same rule as the detector: "motion" when more than
MOVED_FRAC of the pixels move more than PX_LIMIT. Frames go through JET + JPEG
so they load the same way as real captures.

Usage:
    python make_dummy_data.py [--out dummy_data] [--no-eval]
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np

from loftr_motion import MOVED_FRAC, PX_LIMIT, MotionDetector, load_frame

H, W = 512, 640
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
CX, CY = W / 2, H / 2


def shift(dx, dy):
    return np.full((H, W), dx, np.float32), np.full((H, W), dy, np.float32)


def rotate(deg):
    a = np.deg2rad(deg)
    return ((XX - CX) * np.cos(a) - (YY - CY) * np.sin(a) + CX - XX,
            (XX - CX) * np.sin(a) + (YY - CY) * np.cos(a) + CY - YY)


def blob(dx, dy, x, y, sigma):
    """Soft-tissue style local move: full shift at (x, y), fading out with a Gaussian."""
    g = np.exp(-((XX - x) ** 2 + (YY - y) ** 2) / (2 * sigma ** 2))
    return dx * g, dy * g


# (name, flow, gain, offset). gain/offset fake thermal drift: no geometry change.
CASES = [
    ("still_a", shift(0, 0), 1, 0),
    ("still_b", shift(0, 0), 1, 0),
    ("still_c", shift(0, 0), 1, 0),
    ("thermal_drift", shift(0, 0), 1.05, 0.02),
    ("shift_0.5px", shift(0.5, 0), 1, 0),
    ("shift_1px", shift(1, 0), 1, 0),
    ("shift_2px", shift(1.4, 1.4), 1, 0),
    ("shift_3px", shift(3, 0), 1, 0),
    ("shift_5px", shift(4, 3), 1, 0),
    ("shift_10px", shift(0, 10), 1, 0),
    ("rotate_0.2deg", rotate(0.2), 1, 0),
    ("rotate_1deg", rotate(1), 1, 0),
    ("local_small", blob(4, 2, 480, 300, 60), 1, 0),
    ("local_large", blob(6, 4, 480, 300, 150), 1, 0),
]


def render(ref, flow, gain, offset, rng):
    dx, dy = (f.astype(np.float32) for f in flow)
    # cur(x) = ref(x - d): fine as an inverse map for small, smooth flows.
    cur = cv2.remap(ref, XX - dx, YY - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cur = cur * gain + offset + rng.normal(0, 0.01, cur.shape)
    return np.clip(cur, 0, 1).astype(np.float32)


def expected(flow):
    frac = float((np.hypot(*flow) > PX_LIMIT).mean())
    return ("motion" if frac > MOVED_FRAC else "still"), frac


def save_jet(img, path):
    cv2.imwrite(path, cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET),
                [cv2.IMWRITE_JPEG_QUALITY, 95])


def make(out):
    rng = np.random.default_rng(0)
    rows = []
    for src in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pictures", "*.jpg"))):
        ref = load_frame(src)
        assert ref.shape == (H, W), f"{src}: {ref.shape}"
        d = os.path.join(out, os.path.splitext(os.path.basename(src))[0])
        os.makedirs(d, exist_ok=True)
        save_jet(ref, os.path.join(d, "ref.jpg"))
        for i, (name, flow, gain, offset) in enumerate(CASES):
            path = os.path.join(d, f"{i:02d}_{name}.jpg")
            save_jet(render(ref, flow, gain, offset, rng), path)
            want, frac = expected(flow)
            rows.append({"ref": os.path.join(d, "ref.jpg"), "frame": path, "case": name,
                         "true_moved_fraction": round(frac, 3), "expected": want})
    with open(os.path.join(out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} frames to {out}")
    return rows


def evaluate(rows):
    det = MotionDetector()
    refs, wrong = {}, []
    for r in rows:
        if r["ref"] not in refs:
            refs[r["ref"]] = load_frame(r["ref"])
        got = det.check(refs[r["ref"]], load_frame(r["frame"]))
        ok = got["status"] == r["expected"]
        print(f"{'ok ' if ok else 'BAD'} {os.path.relpath(r['frame']):45s} want={r['expected']:6s} "
              f"got={got['status']:14s} moved={got['moved_fraction']:.2f} (true {r['true_moved_fraction']:.2f}) "
              f"n={got['matches']}")
        if not ok:
            wrong.append(r)
    missed = sum(r["expected"] == "motion" for r in wrong)
    print(f"\n{len(rows) - len(wrong)}/{len(rows)} correct; "
          f"{missed} missed motion, {len(wrong) - missed} false alarm / low_confidence")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dummy_data")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    rows = make(args.out)
    if not args.no_eval:
        evaluate(rows)
