# Replays a saved calibration run without the camera.
#
#   python replay.py runs/still_01
#   python replay.py runs/still_01 --windows 0.25 0.5 1.0 2.0
#   python replay.py runs/still_01 --stride 3        (faster, coarser)
#
# Answers two questions the live recorder cannot, because both need the same
# frames measured more than one way:
#
#   - which burst window keeps the gate quietest, which sets BURST_SECONDS
#   - how close phaseCorrelate's confidence gets to MIN_CONFIDENCE, which says
#     whether that guard is set anywhere near the right place
#
# Needs a run recorded WITHOUT --metrics-only, so the .npy frames are on disk.

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_detection as md

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def frame_stamps(folder, paths, fps):
    """Seconds for each frame, from metrics.csv when it is there.

    The recorder writes one CSV row per frame after the first, carrying the
    real elapsed time. Falling back to a flat fps is fine for a rough sweep but
    will smear the window edges if the camera stuttered.
    """
    csv_path = os.path.join(folder, "metrics.csv")
    if os.path.exists(csv_path):
        stamps = {}
        with open(csv_path) as handle:
            next(handle, None)
            for line in handle:
                parts = line.split(",")
                if len(parts) >= 2:
                    stamps[int(parts[0])] = float(parts[1])
        if stamps:
            first = min(stamps) - 1
            stamps[first] = max(0.0, min(stamps.values()) - 1.0 / fps)
            return [stamps.get(i, i / fps) for i in range(len(paths))]

    print(f"  no metrics.csv - assuming a flat {fps} fps")
    return [i / fps for i in range(len(paths))]


def sweep(folder, windows, fps, stride):
    paths = sorted(glob.glob(os.path.join(folder, "f_*.npy")))
    if not paths:
        raise SystemExit(f"no f_*.npy frames in {folder}\n"
                         "record without --metrics-only to keep them")

    stamps = frame_stamps(folder, paths, fps)
    paths, stamps = paths[::stride], stamps[::stride]
    print(f"{len(paths)} frames spanning {stamps[-1] - stamps[0]:.1f}s")

    widest = max(windows)
    history = []                       # (stamp, frame) back as far as we need
    results = {w: [] for w in windows}

    for position, (stamp, path) in enumerate(zip(stamps, paths)):
        frame = np.load(path)
        history.append((stamp, frame))
        while len(history) >= 2 and history[1][0] <= stamp - widest:
            history.pop(0)

        for window in windows:
            # The newest frame that is still a full window behind this one.
            reference = None
            for entry in history:
                if entry[0] <= stamp - window:
                    reference = entry
                else:
                    break
            if reference and stamp - reference[0] >= window * 0.9:
                results[window].append(md.compare(reference[1], frame))

        if position % 100 == 0:
            print(f"  {position}/{len(paths)}", end="\r", flush=True)

    print(" " * 30, end="\r")
    return results


def report(results):
    print(f"\n{'window':>8}{'IoU med':>10}{'IoU w5%':>10}{'IoU worst':>11}"
          f"{'shift w5%':>11}{'shift worst':>13}{'conf min':>11}{'low conf':>10}")

    for window in sorted(results):
        pairs = results[window]
        if not pairs:
            print(f"{window:>8.2f}   (run is shorter than this window)")
            continue

        ious = np.array([p.iou for p in pairs])
        shifts = np.array([p.shift_px for p in pairs])
        confidences = np.array([p.confidence for p in pairs])
        low = int((confidences < md.MIN_CONFIDENCE).sum())

        print(f"{window:>8.2f}{np.median(ious):>10.3f}"
              f"{np.percentile(ious, 5):>10.3f}{ious.min():>11.3f}"
              f"{np.percentile(shifts, 95):>11.2f}{shifts.max():>13.2f}"
              f"{confidences.min():>11.4f}{low:>10}")

    print("\nwindow      burst span being simulated, in seconds")
    print("IoU w5%     the 5th percentile - set MIN_IOU below this on a still run")
    print("shift w5%   the 95th percentile - set MAX_SHIFT_PX above this")
    print(f"low conf    pairs scoring under MIN_CONFIDENCE ({md.MIN_CONFIDENCE}),")
    print("            where the shift was thrown away as unreliable")
    print("\nA shorter window that holds its IoU is the better BURST_SECONDS:")
    print("less time for the patient to drift inside one burst.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="a run folder, e.g. runs/still_01")
    parser.add_argument("--windows", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0],
                        help="burst spans to simulate, in seconds")
    parser.add_argument("--fps", type=float, default=29.0,
                        help="fallback frame rate when metrics.csv is missing")
    parser.add_argument("--stride", type=int, default=1,
                        help="use every Nth frame; 3 is ~3x faster and plenty")
    args = parser.parse_args()

    report(sweep(args.folder, sorted(args.windows), args.fps, max(1, args.stride)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
