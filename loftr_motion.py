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

# ---------------------------------------------------------------------------
# Tuning limits.
#
# These values are fixed. Tune them with the phantom tests (see selftest()).
# A lower limit makes the detector more sensitive, but it also gives more
# false alarms. Change one value at a time and run the selftest again.
# ---------------------------------------------------------------------------

# LoFTR gives each match a confidence between 0 and 1. Matches below this
# value are often wrong on IR frames, because the IR texture is weak.
CONF_MIN = 0.5

# Stop mode: a match is "moved" when it is farther than this distance (in
# pixels) from the position that the camera motion predicts. LoFTR has a
# small sub-pixel error, so do not set this value below 1.
PX_LIMIT = 2.0

# Stop mode: flag motion when this fraction of the matches in one tile moved.
# We use a tile, not the full image, because a small motion (for example, a
# hand) changes only a small part of the image. A global fraction hides it.
MOVED_FRAC = 0.15

# If a pair has fewer matches than this, the result is not reliable.
# verdict() then reports "low_confidence" and never "still".
MIN_MATCHES = 50

# The side of a square image tile, in pixels. A smaller tile finds smaller
# motions, but each tile has fewer matches, so it is less stable.
TILE_PX = 128

# A tile with fewer matches than this is ignored. One or two wrong matches
# in an empty tile must not trigger an alarm.
MIN_TILE_MATCHES = 20

# Scan mode: a match is "moved" when it is farther than this distance (in
# pixels) from its epipolar line. This is larger than PX_LIMIT because the
# fundamental matrix is an estimate and has its own error.
EPI_PX = 5.0

# Scan mode: use only matches on pixels warmer than this value (on the
# normalized 0 to 1 scale). Warm pixels are the body. Cold pixels are the room.
WARM_MIN = 0.5

# Scan mode: drop matches that are nearer than this distance (in pixels) to
# the body outline. When the view turns, the outline is a silhouette, not a
# fixed surface point, so it slides and gives false "moved" matches.
EDGE_PX = 10

# Scan mode: flag motion when this fraction of matches in one tile moved.
# This is higher than MOVED_FRAC, because the epipolar test is more noisy.
SCAN_MOVED_FRAC = 0.3

# A look-up table of the 256 JET colors, in BGR order (OpenCV order).
# Row i is the color that JET gives to the gray value i.
# jet_to_gray() uses it to find the gray value for each color.
# We use int32 so that the squared color differences do not overflow.
_JET_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET).reshape(256, 3).astype(np.int32)


def load_frame(path):
    """Return a float32 frame in [0, 1] from a temperature CSV, a gray image or a JET image."""
    if path.endswith(".csv"):
        # A CSV holds one temperature for each pixel. We scale the frame so
        # that its coldest pixel is 0 and its warmest pixel is 1.
        # Note: each frame gets its own scale. If the warmest point changes
        # between two frames, the same temperature gets a different value.
        temp = np.loadtxt(path, delimiter=",", dtype=np.float32)

        return cv2.normalize(temp, None, 0, 1, cv2.NORM_MINMAX)

    bgr = cv2.imread(path)

    # cv2.imread does not raise an error for a bad path. It returns None.
    if bgr is None:
        raise FileNotFoundError(path)

    # In a gray image that was saved as color, the three channels are equal.
    # We compare blue to red only. A real JET image does not have blue equal
    # to red in all pixels, so this test is enough. Then we use the blue channel.
    if (bgr[..., 0] == bgr[..., 2]).all():
        return bgr[..., 0].astype(np.float32) / 255

    return jet_to_gray(bgr)


def jet_to_gray(bgr):
    """Invert the JET colormap. Plain grayscale of JET is not monotonic in temperature."""
    # Why we cannot use cv2.cvtColor(bgr, COLOR_BGR2GRAY): JET goes from blue
    # to cyan to yellow to red. Yellow and cyan are bright, but red is dark.
    # Thus a hot red pixel gets a lower gray value than a warm yellow pixel.
    # LoFTR would see false edges. We find the true JET index of each color.
    #
    # An image has far fewer distinct colors than pixels. Thus we find the
    # distinct colors first ("colors") and compare only those to the LUT.
    # "inverse" tells us which distinct color each pixel has.
    colors, inverse = np.unique(bgr.reshape(-1, 3), axis=0, return_inverse=True)

    # The squared distance from each distinct color to each of the 256 JET
    # colors. The shape is (number of colors, 256). We use the nearest JET
    # color, because JPEG compression changes the colors a small quantity.
    dist = ((colors[:, None, :].astype(np.int32) - _JET_LUT[None]) ** 2).sum(-1)

    # argmin gives the JET index (0 to 255) for each distinct color.
    # "inverse" maps these indices back to the pixels. Then we divide by 255
    # to get the 0 to 1 scale.
    return (dist.argmin(1)[inverse.ravel()].reshape(bgr.shape[:2]) / 255.0).astype(np.float32)


def worst_tile(xy, moved):
    """Return the highest moved fraction of the tiles that have enough points."""
    if not len(xy):
        return 0.0

    # Integer division by TILE_PX gives the tile (column, row) of each point.
    # np.unique then gives each occupied tile one number, and "tile" holds
    # that number for each point. Empty tiles do not get a number.
    _, tile = np.unique((xy // TILE_PX).astype(np.int32), axis=0, return_inverse=True)

    # The number of points in each tile.
    counts = np.bincount(tile.ravel())

    # The number of moved points in each tile, divided by the total in the tile.
    # "moved" is boolean, so the weights count the moved points.
    fracs = np.bincount(tile.ravel(), weights=moved) / counts

    # Use only the tiles with enough points. If no tile has enough points,
    # max() returns 0 (from "initial"). verdict() then gives "low_confidence"
    # or "still", based on the total number of matches.
    return float(fracs[counts >= MIN_TILE_MATCHES].max(initial=0.0))


class MotionDetector:
    def __init__(self, weights="indoor"):
        # Use the GPU when one is available. LoFTR is slow on a CPU (several
        # seconds for each pair), but the results are the same.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # "indoor" weights were trained on ScanNet (indoor rooms). "outdoor"
        # weights were trained on MegaDepth. Indoor is nearer to our setup.
        # eval() sets the model to inference mode (no dropout, fixed norms).
        self.matcher = KF.LoFTR(pretrained=weights).eval().to(self.device)

    def match(self, ref, cur):
        """Return matched points (N, 2) in ref and cur."""
        # LoFTR needs sides that are a multiple of 8, because its coarse
        # level works on 1/8 of the resolution. We crop the right and bottom
        # edges. We do not resize, because a resize changes the pixel scale.
        h, w = (ref.shape[0] // 8) * 8, (ref.shape[1] // 8) * 8

        # Change an (H, W) numpy array to a (1, 1, H, W) tensor: one image,
        # one gray channel. ascontiguousarray is necessary because the crop
        # makes a view that torch.from_numpy cannot always use.
        t = lambda img: torch.from_numpy(np.ascontiguousarray(img[:h, :w]))[None, None].to(self.device)

        # inference_mode turns off the gradient records, which saves memory and time.
        with torch.inference_mode():
            out = self.matcher({"image0": t(ref), "image1": t(cur)})

        # Remove the matches with a low confidence.
        keep = out["confidence"] > CONF_MIN

        # keypoints0[i] in ref matches keypoints1[i] in cur. Each row is (x, y) in pixels.
        return out["keypoints0"][keep].cpu().numpy(), out["keypoints1"][keep].cpu().numpy()

    def check(self, ref, cur, expected_shift=(0.0, 0.0)):
        """Compare cur to ref.

        expected_shift is the image shift (dx, dy) in pixels that the camera
        motion causes, from the hardware angle. Use (0, 0) at a capture stop.
        """
        p0, p1 = self.match(ref, cur)

        # For a still scene, each match moves by expected_shift only.
        # The residual is the remaining motion of each match, in pixels.
        # Note: one shift for all points is correct only for a small camera
        # motion. For a large turn, use check_scan(), which uses epipolar geometry.
        residual = np.linalg.norm(p1 - p0 - np.asarray(expected_shift), axis=1)

        return verdict(p0, p1, residual, PX_LIMIT)

    def check_scan(self, frames):
        """Check a full rotation. frames[i] and frames[i + 1] are one click apart, and the last wraps to the first.

        Each click is the same turn, so one fundamental matrix F fits every pair of a still scene.
        Estimate F from all pairs together, so a patient motion in a few pairs is only a set of outliers.
        A match that is far from its epipolar line moved. Returns one result for each pair.
        Blind spot: a motion along the epipolar lines (about horizontal here) does not show.
        """
        # Theory:
        # The fundamental matrix F relates the two views of one rigid scene.
        # For a point x0 in the first view, F @ x0 is a line in the second
        # view (the epipolar line). The match x1 of a still point is on that line.
        # If the patient moves, x1 goes off the line, and the distance shows it.
        #
        # Why one F for all pairs: the camera turns the same angle at each
        # click, so the geometry between frame i and frame i + 1 is the same
        # for all i. We put the matches of all 48 pairs into one estimate.
        # A patient motion affects only one or two pairs, so it is a small
        # part of the data. The robust estimator rejects it as outliers.
        pairs = []

        # A square kernel of side 2 * EDGE_PX + 1. Erosion with this kernel
        # removes a band of EDGE_PX pixels along the body outline.
        kernel = np.ones((2 * EDGE_PX + 1, 2 * EDGE_PX + 1), np.uint8)

        for i, ref in enumerate(frames):
            # The modulo makes the last frame pair with the first frame,
            # because the rotation is a closed circle.
            p0, p1 = self.match(ref, frames[(i + 1) % len(frames)])

            # The room does not follow the rotation in the phantom setup, so use only the warm body.
            # The room matches obey a different geometry, so they would
            # pull F away from the body geometry.
            body = cv2.erode((ref > WARM_MIN).astype(np.uint8), kernel)

            # Look up the mask at each ref point. The mask is indexed [row, column],
            # thus [y, x]. LoFTR points are (x, y), so we swap the columns.
            keep = body[p0[:, 1].astype(int), p0[:, 0].astype(int)] > 0
            pairs.append((p0[keep], p1[keep]))

        # Estimate F from the matches of all pairs together.
        # USAC_MAGSAC is a robust estimator. It is better than plain RANSAC
        # when the inlier threshold is not known exactly.
        # 1.0 is the inlier threshold (pixels), 0.9999 is the confidence and
        # 100000 is the maximum number of iterations. The high values cost
        # time, but we estimate F only one time for each scan.
        F, _ = cv2.findFundamentalMat(np.vstack([p[0] for p in pairs]), np.vstack([p[1] for p in pairs]),
                                      cv2.USAC_MAGSAC, 1.0, 0.9999, 100000)
        results = []

        for p0, p1 in pairs:
            # For each ref point, get its epipolar line (a, b, c) in the next
            # frame. The line is a*x + b*y + c = 0. The "1" means that the
            # points are in the first image. A pair with no matches gets an
            # empty array, because computeCorrespondEpilines fails on empty input.
            # OpenCV normalizes the lines, so this is the distance in pixels.
            lines = cv2.computeCorrespondEpilines(p0[:, None], 1, F).reshape(-1, 3) if len(p0) else np.zeros((0, 3))

            # With a^2 + b^2 = 1, |a*x + b*y + c| is the distance from the
            # point (x, y) to the line, in pixels.
            dist = np.abs((lines[:, :2] * p1).sum(1) + lines[:, 2])
            results.append(verdict(p0, p1, dist, EPI_PX, SCAN_MOVED_FRAC))

        return results


def verdict(p0, p1, residual, limit, frac_limit=MOVED_FRAC):
    """Flag motion when more than frac_limit of the matches in one tile have a residual above limit."""
    # A boolean flag for each match: True if the match moved too far.
    moved = residual > limit

    # The global moved fraction. We report it, but we do not use it for the decision.
    frac = float(moved.mean()) if len(p0) else 0.0

    # The local moved fraction. This is the value for the decision.
    tile_frac = worst_tile(p0, moved)

    if len(p0) < MIN_MATCHES:
        # Fail safe: too few matches is never reported as "still".
        # For a patient scan, a missed motion is worse than a false alarm.
        status = "low_confidence"
    else:
        status = "motion" if tile_frac > frac_limit else "still"

    # The result dict. p0, p1 and moved are for draw(). The other fields are for the report.
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
    # BONE is a gray-blue colormap. It keeps the background quiet, so the
    # red and green arrows are easy to see.
    img = cv2.applyColorMap((cur * 255).astype(np.uint8), cv2.COLORMAP_BONE)

    # Each arrow goes from the ref position to the cur position of one match.
    # OpenCV colors are BGR, thus (0, 0, 255) is red.
    for a, b, m in zip(result["p0"], result["p1"], result["moved"]):
        color = (0, 0, 255) if m else (0, 200, 0)
        cv2.arrowedLine(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, 1, tipLength=0.3)

    # Write the result on the top left corner of the image.
    cv2.putText(img, f"{result['status']}  moved={result['moved_fraction']:.0%}  n={result['matches']}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(path, img)


def scan_frames(folder):
    """Return the frame paths in folder in click order (IR_1, IR_2, ..., IR_48)."""
    # Keep only the files whose name ends with a number and an image or CSV extension.
    names = [n for n in os.listdir(folder) if re.search(r"\d+\.(jpg|png|csv)$", n)]

    # Sort by the last number in the name, as an integer. A text sort would
    # put IR_10 before IR_2, which breaks the click order.
    return [os.path.join(folder, n) for n in sorted(names, key=lambda n: int(re.findall(r"\d+", n)[-1]))]


def selftest():
    """Run scan mode on the still phantom scan, with and without a synthetic local motion."""
    # The test has two parts:
    # 1. The real still scan must give no alarms (no false positives).
    # 2. The same scan with a fake motion in one frame must give an alarm
    #    at that frame (no false negative).
    here = os.path.dirname(os.path.abspath(__file__))
    frames = [load_frame(p) for p in scan_frames(os.path.join(here, "pictures", "new"))]
    det = MotionDetector()

    # Return the 1-based numbers of the pairs that are not "still".
    # "low_confidence" also counts as a flag, because of the fail safe.
    flagged = lambda results: [i + 1 for i, r in enumerate(results) if r["status"] != "still"]

    # Make a copy of the frame list. Only frame 25 changes, and we copy it
    # so that "frames" stays the same for the still test.
    moved = list(frames)

    # Shift the right half of frame 25 down by 3 px. This makes pairs 24->25 and 25->26 wrong.
    # The 2x3 matrix [[1, 0, 0], [0, 1, 3]] is a pure translation of 3 px in y.
    # A vertical shift is across the epipolar lines (about horizontal), so
    # check_scan can see it. A horizontal shift would be in its blind spot.
    # BORDER_REPLICATE fills the empty top rows with the edge pixels, so no
    # black band (a false edge) appears.
    moved[24] = moved[24].copy()
    moved[24][:, 320:] = cv2.warpAffine(frames[24], np.float32([[1, 0, 0], [0, 1, 3]]), frames[24].shape[::-1],
                                        borderMode=cv2.BORDER_REPLICATE)[:, 320:]

    still = flagged(det.check_scan(frames))
    found = flagged(det.check_scan(moved))
    print(f"still scan flagged pairs: {still}\nscan with motion at frame 25 flagged pairs: {found}")

    assert still == [], f"false alarms: {still}"

    # Pair 24 is frame 24 -> frame 25, the first pair that sees the fake motion.
    assert 24 in found, f"missed the motion: {found}"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # ref and cur are optional here, because --scan and --selftest do not use them.
    # We check them by hand below.
    ap.add_argument("ref", nargs="?")
    ap.add_argument("cur", nargs="*")
    ap.add_argument("--shift", nargs=2, type=float, default=(0, 0), metavar=("DX", "DY"))
    ap.add_argument("--out", help="folder for match images")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scan", metavar="DIR", help="check a full rotation in DIR (IR_1, IR_2, ...)")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    # Scan mode: check each pair of neighbor frames in a full rotation.
    if args.scan:
        paths = scan_frames(args.scan)
        frames = [load_frame(p) for p in paths]

        for i, r in enumerate(MotionDetector().check_scan(frames)):
            nxt = paths[(i + 1) % len(paths)]
            print(f"{os.path.basename(paths[i])} -> {os.path.basename(nxt)}: {r['status']} "
                  f"tile={r['worst_tile_fraction']:.2f} n={r['matches']} median_px={r['median_residual_px']}")

            if args.out:
                os.makedirs(args.out, exist_ok=True)

                # Draw on the second frame of the pair, because the arrows end there.
                draw(frames[(i + 1) % len(frames)], r, os.path.join(args.out, f"pair_{i + 1}.png"))
        return

    if not args.ref or not args.cur:
        ap.error("give REF and at least one CUR, or --selftest")

    # Stop mode: compare each CUR frame to the same REF frame.
    # We load the model and REF one time only, because both are slow to load.
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
