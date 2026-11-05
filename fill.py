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

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_detection as md


def middle_frame(folder):
    """The middle frame of a position, so one noisy grab cannot skew it."""
    paths = sorted(glob.glob(os.path.join(folder, "f_*.npy")))
    if not paths:
        return None
    return np.load(paths[len(paths) // 2])


def report(folder):
    frame = middle_frame(folder)
    if frame is None:
        return
    mask = md.body_mask(frame)
    edge = bool(mask[0].any() or mask[-1].any()
                or mask[:, 0].any() or mask[:, -1].any())
    print(f"{os.path.basename(folder):>12}{100 * mask.mean():8.0f}%"
          f"   {'TOUCHES EDGE' if edge else ''}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python fill.py <folder of positions>")

    for target in sys.argv[1:]:
        folders = sorted(d for d in glob.glob(os.path.join(target, "*"))
                         if os.path.isdir(d))
        if not folders:                      # a single position folder
            folders = [target]
        print(f"\n{target}")
        print(f"{'position':>12}{'fill':>9}")
        for folder in folders:
            report(folder)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
