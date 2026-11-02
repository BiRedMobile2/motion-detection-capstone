"""Proof of concept: detect patient motion in IR frames with LoFTR.

LoFTR (Sun et al., CVPR 2021, github.com/zju3dv/LoFTR) finds dense
feature matches between two images without a keypoint detector. This is
useful for IR frames, because they have little texture.

Method:
1. Match the reference frame to the current frame with LoFTR.
2. Subtract the expected shift from the camera motion (zero when parked).
3. If too many matches still move more than a pixel limit, flag motion.

Usage:
    python loftr_motion.py REF CUR [CUR ...] [--shift DX DY] [--out DIR]
    python loftr_motion.py --selftest

REF and CUR are JET color images (as IRCameraCapture saves them) or
temperature CSV files (as IRCameraCapture.SaveImages writes them).
"""

import argparse
import os

import cv2
import kornia.feature as KF
import numpy as np
import torch

# fixed limits, tune them with the phantom tests.
CONF_MIN = 0.5      # Keep LoFTR matches with a confidence above this value.
PX_LIMIT = 2.0      # A match that moves more than this (pixels) is "moved".
MOVED_FRAC = 0.15   # Flag motion when this fraction of matches moved.
MIN_MATCHES = 50    # Below this, the result is not reliable.

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
        moved = residual > PX_LIMIT
        frac = float(moved.mean()) if len(p0) else 0.0

        if len(p0) < MIN_MATCHES:
            # Fail safe: too few matches is never reported as "still".
            status = "low_confidence"
        else:
            status = "motion" if frac > MOVED_FRAC else "still"
        return {
            "status": status,
            "matches": len(p0),
            "moved_fraction": round(frac, 3),
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


def selftest():
    """Make synthetic motion from a real IR image and check each verdict."""
    here = os.path.dirname(os.path.abspath(__file__))
    ref = load_frame(os.path.join(here, "pictures", "IR_Test.jpg"))
    rng = np.random.default_rng(0)
    noisy = lambda img: np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1).astype(np.float32)
    shift = lambda img, dx, dy: cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]),
                                               img.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
    local = ref.copy()
    local[:, 320:] = shift(ref, 6, 4)[:, 320:]   # Only the right half (the arm) moves.

    det = MotionDetector()
    cases = [
        ("still + sensor noise", noisy(ref), (0, 0), "still"),
        ("global shift 5px", noisy(shift(ref, 5, 3)), (0, 0), "motion"),
        ("local shift 6px", noisy(local), (0, 0), "motion"),
        ("camera shift, expected", noisy(shift(ref, 8, 0)), (8, 0), "still"),
        ("flat frame", np.full_like(ref, 0.5), (0, 0), "low_confidence"),
    ]
    
    for name, cur, exp, want in cases:
        r = det.check(ref, cur, exp)

        print(f"{name:24s} -> {r['status']:15s} moved={r['moved_fraction']:.2f} "
              f"n={r['matches']} median={r['median_residual_px']}")
        
        assert r["status"] == want, f"{name}: got {r['status']}, want {want}"
    
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?")
    ap.add_argument("cur", nargs="*")
    ap.add_argument("--shift", nargs=2, type=float, default=(0, 0), metavar=("DX", "DY"))
    ap.add_argument("--out", help="folder for match images")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    
    if not args.ref or not args.cur:
        ap.error("give REF and at least one CUR, or --selftest")

    det = MotionDetector()
    ref = load_frame(args.ref)

    for path in args.cur:
        cur = load_frame(path)
        r = det.check(ref, cur, args.shift)

        print(f"{path}: {r['status']} moved={r['moved_fraction']:.2f} "
              f"n={r['matches']} median_px={r["median_residual_px"]}")
        
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            draw(cur, r, os.path.join(args.out, os.path.basename(path).rsplit(".", 1)[0] + "_matches.png"))


if __name__ == "__main__":
    main()
