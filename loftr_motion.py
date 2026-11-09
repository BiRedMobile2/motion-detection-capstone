"""Proof of concept: detect patient motion in IR frames with LoFTR.

LoFTR (Sun et al., CVPR 2021, github.com/zju3dv/LoFTR) finds dense
feature matches between two images without a keypoint detector. This is
useful for IR frames, because they have little texture.

Method:
1. Match the reference frame to the current frame with LoFTR.
2. Subtract the expected shift from the camera motion (zero when parked).
3. If too many matches in any image tile still move more than a pixel limit, flag motion.

Scan mode (--scan) checks a full rotation, where the camera turns one
click between frames. It uses the epipolar geometry of one click in place
of a shift (see MotionDetector.check_scan).

Usage:
    python loftr_motion.py REF CUR [CUR ...] [--shift DX DY] [--out DIR]
    python loftr_motion.py --scan DIR [--out DIR]
    python loftr_motion.py --selftest

REF and CUR are JET color images (as IRCameraCapture saves them) or
temperature CSV files (as IRCameraCapture.SaveImages writes them).
"""

import argparse
import os
import re

import cv2
import kornia.feature as KF
import numpy as np
import torch

# fixed limits, tune them with the phantom tests.
CONF_MIN = 0.5      # Keep LoFTR matches with a confidence above this value.
PX_LIMIT = 2.0      # A match that moves more than this (pixels) is "moved".
MOVED_FRAC = 0.15   # Flag motion when this fraction of matches in one tile moved.
MIN_MATCHES = 50    # Below this, the result is not reliable.
TILE_PX = 128       # Side of a square image tile (pixels).
MIN_TILE_MATCHES = 20  # Ignore a tile with fewer matches than this.
EPI_PX = 5.0        # Scan mode: a match farther than this (pixels) from its epipolar line is "moved".
WARM_MIN = 0.5      # Scan mode: use only matches on pixels warmer than this (0 to 1 scale).
EDGE_PX = 10        # Scan mode: drop matches this near the body outline. The outline slides as the view turns.
SCAN_MOVED_FRAC = 0.3  # Scan mode: flag motion when this fraction of matches in one tile moved.

_JET_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET).reshape(256, 3).astype(np.int32)


def load_frame(path):
    """Return a float32 frame in [0, 1] from a temperature CSV, a gray image or a JET image."""
    if path.endswith(".csv"):
        temp = np.loadtxt(path, delimiter=",", dtype=np.float32)

        return cv2.normalize(temp, None, 0, 1, cv2.NORM_MINMAX)
    
    bgr = cv2.imread(path)

    if bgr is None:
        raise FileNotFoundError(path)
    if (bgr[..., 0] == bgr[..., 2]).all():
        return bgr[..., 0].astype(np.float32) / 255
    
    return jet_to_gray(bgr)


def jet_to_gray(bgr):
    """Invert the JET colormap. Plain grayscale of JET is not monotonic in temperature."""
    colors, inverse = np.unique(bgr.reshape(-1, 3), axis=0, return_inverse=True)
    dist = ((colors[:, None, :].astype(np.int32) - _JET_LUT[None]) ** 2).sum(-1)
    
    return (dist.argmin(1)[inverse.ravel()].reshape(bgr.shape[:2]) / 255.0).astype(np.float32)


def worst_tile(xy, moved):
    """Return the highest moved fraction of the tiles that have enough points."""
    if not len(xy):
        return 0.0

    _, tile = np.unique((xy // TILE_PX).astype(np.int32), axis=0, return_inverse=True)
    counts = np.bincount(tile.ravel())
    fracs = np.bincount(tile.ravel(), weights=moved) / counts

    return float(fracs[counts >= MIN_TILE_MATCHES].max(initial=0.0))


class MotionDetector:
    def __init__(self, weights="indoor"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.matcher = KF.LoFTR(pretrained=weights).eval().to(self.device)

    def match(self, ref, cur):
        """Return matched points (N, 2) in ref and cur."""
        # LoFTR needs sides that are a multiple of 8.
        h, w = (ref.shape[0] // 8) * 8, (ref.shape[1] // 8) * 8
        t = lambda img: torch.from_numpy(np.ascontiguousarray(img[:h, :w]))[None, None].to(self.device)
        
        with torch.inference_mode():
            out = self.matcher({"image0": t(ref), "image1": t(cur)})
        
        keep = out["confidence"] > CONF_MIN

        return out["keypoints0"][keep].cpu().numpy(), out["keypoints1"][keep].cpu().numpy()

    def check(self, ref, cur, expected_shift=(0.0, 0.0)):
        """Compare cur to ref.

        expected_shift is the image shift (dx, dy) in pixels that the camera
        motion causes, from the hardware angle. Use (0, 0) at a capture stop.
        """
        p0, p1 = self.match(ref, cur)
        residual = np.linalg.norm(p1 - p0 - np.asarray(expected_shift), axis=1)

        return verdict(p0, p1, residual, PX_LIMIT)

    def check_scan(self, frames):
        """Check a full rotation. frames[i] and frames[i + 1] are one click apart, and the last wraps to the first.

        Each click is the same turn, so one fundamental matrix F fits every pair of a still scene.
        Estimate F from all pairs together, so a patient motion in a few pairs is only a set of outliers.
        A match that is far from its epipolar line moved. Returns one result for each pair.
        Blind spot: a motion along the epipolar lines (about horizontal here) does not show.
        """
        pairs = []
        kernel = np.ones((2 * EDGE_PX + 1, 2 * EDGE_PX + 1), np.uint8)

        for i, ref in enumerate(frames):
            p0, p1 = self.match(ref, frames[(i + 1) % len(frames)])
            # The room does not follow the rotation in the phantom setup, so use only the warm body.
            body = cv2.erode((ref > WARM_MIN).astype(np.uint8), kernel)
            keep = body[p0[:, 1].astype(int), p0[:, 0].astype(int)] > 0
            pairs.append((p0[keep], p1[keep]))

        F, _ = cv2.findFundamentalMat(np.vstack([p[0] for p in pairs]), np.vstack([p[1] for p in pairs]),
                                      cv2.USAC_MAGSAC, 1.0, 0.9999, 100000)
        results = []

        for p0, p1 in pairs:
            # OpenCV normalizes the lines, so this is the distance in pixels.
            lines = cv2.computeCorrespondEpilines(p0[:, None], 1, F).reshape(-1, 3) if len(p0) else np.zeros((0, 3))
            dist = np.abs((lines[:, :2] * p1).sum(1) + lines[:, 2])
            results.append(verdict(p0, p1, dist, EPI_PX, SCAN_MOVED_FRAC))

        return results


def verdict(p0, p1, residual, limit, frac_limit=MOVED_FRAC):
    """Flag motion when more than frac_limit of the matches in one tile have a residual above limit."""
    moved = residual > limit
    frac = float(moved.mean()) if len(p0) else 0.0
    tile_frac = worst_tile(p0, moved)

    if len(p0) < MIN_MATCHES:
        # Fail safe: too few matches is never reported as "still".
        status = "low_confidence"
    else:
        status = "motion" if tile_frac > frac_limit else "still"
    return {
        "status": status,
        "matches": len(p0),
        "moved_fraction": round(frac, 3),
        "worst_tile_fraction": round(tile_frac, 3),
        "median_residual_px": round(float(np.median(residual)), 2) if len(p0) else None,
        "p0": p0, "p1": p1, "moved": moved,
    }


def draw(cur, result, path):
    """Save cur with match arrows. Red arrows moved, green arrows did not."""
    img = cv2.applyColorMap((cur * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    
    for a, b, m in zip(result["p0"], result["p1"], result["moved"]):
        color = (0, 0, 255) if m else (0, 200, 0)
        cv2.arrowedLine(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, 1, tipLength=0.3)
    
    cv2.putText(img, f"{result['status']}  moved={result['moved_fraction']:.0%}  n={result['matches']}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(path, img)


def scan_frames(folder):
    """Return the frame paths in folder in click order (IR_1, IR_2, ..., IR_48)."""
    names = [n for n in os.listdir(folder) if re.search(r"\d+\.(jpg|png|csv)$", n)]

    return [os.path.join(folder, n) for n in sorted(names, key=lambda n: int(re.findall(r"\d+", n)[-1]))]


def selftest():
    """Run scan mode on the still phantom scan, with and without a synthetic local motion."""
    here = os.path.dirname(os.path.abspath(__file__))
    frames = [load_frame(p) for p in scan_frames(os.path.join(here, "pictures", "new"))]
    det = MotionDetector()

    flagged = lambda results: [i + 1 for i, r in enumerate(results) if r["status"] != "still"]
    moved = list(frames)
    # Shift the right half of frame 25 down by 3 px. This makes pairs 24->25 and 25->26 wrong.
    moved[24] = moved[24].copy()
    moved[24][:, 320:] = cv2.warpAffine(frames[24], np.float32([[1, 0, 0], [0, 1, 3]]), frames[24].shape[::-1],
                                        borderMode=cv2.BORDER_REPLICATE)[:, 320:]

    still = flagged(det.check_scan(frames))
    found = flagged(det.check_scan(moved))
    print(f"still scan flagged pairs: {still}\nscan with motion at frame 25 flagged pairs: {found}")

    assert still == [], f"false alarms: {still}"
    assert 24 in found, f"missed the motion: {found}"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?")
    ap.add_argument("cur", nargs="*")
    ap.add_argument("--shift", nargs=2, type=float, default=(0, 0), metavar=("DX", "DY"))
    ap.add_argument("--out", help="folder for match images")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scan", metavar="DIR", help="check a full rotation in DIR (IR_1, IR_2, ...)")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.scan:
        paths = scan_frames(args.scan)
        frames = [load_frame(p) for p in paths]

        for i, r in enumerate(MotionDetector().check_scan(frames)):
            nxt = paths[(i + 1) % len(paths)]
            print(f"{os.path.basename(paths[i])} -> {os.path.basename(nxt)}: {r['status']} "
                  f"tile={r['worst_tile_fraction']:.2f} n={r['matches']} median_px={r['median_residual_px']}")

            if args.out:
                os.makedirs(args.out, exist_ok=True)
                draw(frames[(i + 1) % len(frames)], r, os.path.join(args.out, f"pair_{i + 1}.png"))
        return

    if not args.ref or not args.cur:
        ap.error("give REF and at least one CUR, or --selftest")

    det = MotionDetector()
    ref = load_frame(args.ref)

    for path in args.cur:
        cur = load_frame(path)
        r = det.check(ref, cur, args.shift)

        print(f"{path}: {r['status']} moved={r['moved_fraction']:.2f} tile={r['worst_tile_fraction']:.2f} "
              f"n={r['matches']} median_px={r["median_residual_px"]}")
        
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            draw(cur, r, os.path.join(args.out, os.path.basename(path).rsplit(".", 1)[0] + "_matches.png"))


if __name__ == "__main__":
    main()
