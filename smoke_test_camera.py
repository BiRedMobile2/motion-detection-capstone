# First run against the real camera. Answers the questions the stubbed tests
# cannot: what the frames actually contain, whether they are oriented the way
# we assume, and whether a burst gets distinct frames.
#
#   python smoke_test_camera.py
#
# Needs the camera, the DLLs and numpy/OpenCV. No Qt, so it runs even if
# PySide6 is not set up yet. Writes two files you should look at afterwards.

import ctypes as ct
import os
import time

import cv2
import numpy as np

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SKIN_TEMP_C = 28.0          # the value motion_detection.py assumes
BURST_FRAMES = 5
BURST_SECONDS = 1.0


def main():
    print("loading DLLs...")
    ct.cdll.LoadLibrary(os.path.join(THIS_DIR, "opencv_core440.dll"))
    dll = ct.cdll.LoadLibrary(os.path.join(THIS_DIR, "icisdk_x64.dll"))

    print("starting camera...")
    dll.StartCamera(0, 1)
    try:
        return run_checks(dll)
    finally:
        # Always hand the camera back. Leaving it open makes every later run
        # fail the same way, and you can hear the shutter still going.
        print("stopping camera...")
        dll.StopCamera()


def run_checks(dll):
    height = dll.GetImgHeight()
    width = dll.GetImgWidth()
    print(f"  reported size: {height} x {width}")
    if height <= 0 or width <= 0:
        print("  !! camera returned a bad size - nothing else will work")
        print("     if the shutter is still clicking after this exits, something")
        print("     else has the camera open: close it, or replug the USB.")
        return 1

    dll.GetTemperatureImage.restype = ct.POINTER(ct.c_float)

    def grab():
        pointer = dll.GetTemperatureImage()
        if not pointer:
            return None
        raw = np.ctypeslib.as_array(pointer, shape=(height, width)).copy()
        flipped = cv2.flip(raw, 1)
        return cv2.rotate(flipped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Let the camera settle before judging anything.
    for _ in range(10):
        grab()
        time.sleep(0.05)

    # ---- 1. what is actually in a frame -----------------------------------
    print("\n1. frame contents")
    frame = grab()
    if frame is None:
        print("  !! GetTemperatureImage returned NULL")
        return 1

    print(f"  shape      {frame.shape}   (after flip + rotate)")
    print(f"  dtype      {frame.dtype}")
    print(f"  min        {frame.min():.2f}")
    print(f"  max        {frame.max():.2f}")
    print(f"  mean       {frame.mean():.2f}")
    print(f"  median     {np.median(frame):.2f}")

    if 10 < frame.min() < 40 and 20 < frame.max() < 50:
        print("  -> looks like degrees C")
    else:
        print("  -> NOT the range we expected for degrees C. SKIN_TEMP_C will")
        print("     be wrong, and body_mask needs the Otsu path instead.")

    # ---- 2. does the 28 C cut separate anything ---------------------------
    print(f"\n2. the {SKIN_TEMP_C} C threshold")
    above = float((frame >= SKIN_TEMP_C).mean())
    print(f"  {above * 100:.1f}% of pixels are at or above {SKIN_TEMP_C} C")
    if above < 0.02:
        print("  -> almost nothing is warm. Stand in frame, or the cut is too high.")
    elif above > 0.90:
        print("  -> almost everything is warm. The cut is too low for this scene.")
    else:
        print("  -> plausible: part body, part background.")

    # ---- 3. how fast can we read ------------------------------------------
    print("\n3. read speed")
    started = time.perf_counter()
    reads = 20
    for _ in range(reads):
        grab()
    per_frame = (time.perf_counter() - started) / reads
    print(f"  {per_frame * 1000:.1f} ms per read  ({1 / per_frame:.1f} reads/s)")

    # ---- 4. does a burst get DIFFERENT frames -----------------------------
    # If the camera is slower than the burst pacing, consecutive reads return
    # the same frame. The motion check would then compare a frame to itself,
    # score a perfect match, and pass everything.
    print("\n4. burst frames are distinct")
    frames = []
    interval = BURST_SECONDS / max(BURST_FRAMES - 1, 1)
    while len(frames) < BURST_FRAMES:
        at = time.perf_counter()
        f = grab()
        if f is not None:
            frames.append(f)
        left = interval - (time.perf_counter() - at)
        if left > 0:
            time.sleep(left)

    duplicates = 0
    for i in range(1, len(frames)):
        if np.array_equal(frames[i], frames[i - 1]):
            duplicates += 1
    print(f"  {len(frames)} frames, {duplicates} identical to the one before")
    if duplicates:
        print("  -> DUPLICATES. Raise BURST_SECONDS or lower BURST_FRAMES, or")
        print("     every burst will look perfectly still.")
    else:
        print("  -> all distinct.")

    # ---- 5. leave something to look at ------------------------------------
    print("\n5. files for you to check")
    npy_path = os.path.join(THIS_DIR, "smoke_frame.npy")
    jpg_path = os.path.join(THIS_DIR, "smoke_frame.jpg")
    np.save(npy_path, frame)

    eight_bit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    colored = cv2.applyColorMap(cv2.cvtColor(eight_bit, cv2.COLOR_GRAY2RGB),
                                cv2.COLORMAP_JET)
    cv2.imwrite(jpg_path, colored)
    print(f"  {npy_path}")
    print(f"  {jpg_path}")
    print("  -> open the jpg. Is it the right way up, and the same way round")
    print("     as the visible camera? If not, fix the flip/rotate in")
    print("     IRCameraCapture.ReadTemperatureFrame.")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
