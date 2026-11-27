# Reports what fraction of the frame the body mask covers, per position.
#
#   python fill.py grabs/s1          every position in that folder
#   python fill.py grabs/s1/p00      a single position
#
# Fill fraction is the number that makes a rotation result transferable.
# Outline overlap is a ratio, so sensitivity depends on how much of the frame
# the subject occupies rather than on how far away the camera is standing.
#
# TOUCHES EDGE means part of the mask's outline is the image border rather
# than the subject, which inflates the overlap and makes real rotation look
# smaller than it is. A run has already been invalidated that way.
#
# stray is the share of the mask that is not its largest piece: a warm
# turntable deck, a lamp, a hand. It does not move when the subject rotates, so
# it props the overlap up, and the effect grows as coverage falls. Anything
# above about 1 percent is worth tracking down before measuring.

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_detection as md


def middle_frame(folder):
    """The middle frame of a position, so one noisy grab cannot skew it."""
    paths = sorted(glob.glob(os.path.join(folder, "f_*.npy")))
    if not paths:
        return None
    return np.load(paths[len(paths) // 2])


def report(folder, span=False):
    frame = middle_frame(folder)
    if frame is None:
        return
    mask = md.body_mask(frame)
    edge = bool(mask[0].any() or mask[-1].any()
                or mask[:, 0].any() or mask[:, -1].any())
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
    stray = sum(areas[1:]) / sum(areas) if areas else 0.0
    line = (f"{os.path.basename(folder):>12}{100 * mask.mean():9.0f}%"
            f"{count - 1:8d}{100 * stray:9.1f}%")
    if span:
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size:
            line += f"{rows.max() - rows.min() + 1:9d}{cols.max() - cols.min() + 1:9d}"
        else:
            line += f"{'-':>9}{'-':>9}"
    print(line + ("   TOUCHES EDGE" if edge else ""))


def main():
    parser = argparse.ArgumentParser(
        description="What fraction of the frame the body mask covers.")
    parser.add_argument("targets", nargs="+",
                        help="a folder of positions, or one position folder")
    parser.add_argument("--span", action="store_true",
                        help="also print the mask's height and width in pixels")
    args = parser.parse_args()

    for target in args.targets:
        folders = sorted(d for d in glob.glob(os.path.join(target, "*"))
                         if os.path.isdir(d))
        if not folders:                      # a single position folder
            folders = [target]
        print(f"\n{target}")
        header = f"{'position':>12}{'coverage':>10}{'blobs':>8}{'stray':>10}"
        if args.span:
            header += f"{'rows px':>9}{'cols px':>9}"
        print(header)
        for folder in folders:
            report(folder, args.span)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
