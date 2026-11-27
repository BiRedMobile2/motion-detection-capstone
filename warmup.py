# Prints how warm the subject is compared to the room, for watching a phantom
# settle before a drift run.
#
#   python warmup.py grabs/aim/a
#
# Contrast is the subject's median minus the room's median, in raw counts.
# While it is still climbing the subject has not reached equilibrium, and a run
# started now rejects on a growing mask rather than on motion.
#
# It is the median rather than the maximum on purpose. The hottest pixel is the
# heat source itself, which is a point and moves around with noise. The median
# over the whole mask is what the outline actually depends on.

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Subject contrast over room, for watching a phantom settle.")
    parser.add_argument("folder", help="a position folder, such as grabs/aim/a")
    args = parser.parse_args()

    frame = middle_frame(args.folder)
    if frame is None:
        raise SystemExit(f"no frames in {args.folder}")

    frame = frame.astype(np.float32)
    mask = md.body_mask(frame)
    room = float(np.median(frame[~mask]))
    subject = float(np.median(frame[mask]))

    print(f"  coverage   {100 * mask.mean():.1f}%")
    print(f"  room       {room:.0f}")
    print(f"  subject    {subject:.0f}")
    print(f"  contrast   {subject - room:.0f} counts over room")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
