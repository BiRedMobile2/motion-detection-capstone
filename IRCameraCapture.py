# Class to interact with the IR camera, sending images to UI and saving to folder.
# Adapted from Carlos' code
# Scott N. 9/26/2022
#
# Burst capture with a patient-motion gate added. The live feed behaves as
# before; CapturePosition() and CaptureClosure() are new.
#
# All SDK access stays on this QThread. A capture is queued with a flag the run
# loop picks up, rather than called straight through from the GUI thread - two
# threads calling GetTemperatureImage() at once would hand back overlapping
# views of the same SDK buffer.

import os
import threading
import time

import cv2
import ctypes as ct
import numpy as np
from PySide6.QtCore import *
from PySide6.QtWidgets import QWidget, QLabel
from  PySide6.QtGui import QImage, QPixmap

import motion_detection as md

# Frames per burst and how long to spread them over.
#
# A second, measured. A shorter window does give the patient less time to drift
# inside one burst, but it shrinks real movement by exactly the same mechanism,
# and movement is what we are trying to see. Worst-5% IoU on a run with
# deliberate small shoulder shifts:
#
#   0.25s  0.944      0.5s  0.848      1.0s  0.712      2.0s  0.633
#
# Against a breathing volunteer's 0.987 at the same window, 1 second separates
# movement from breathing about twice as well as half a second does. 2 seconds
# separates better still, but leaves more of the burst exposed to drift for a
# gain that is already past the threshold.
BURST_FRAMES = 5
BURST_SECONDS = 1.0

# Seconds to wait after the rig moves, before capturing, so gantry vibration
# dies down. PLACEHOLDER - measure this on the real rig.
SETTLE_SECONDS = 0.5

MAX_RETRIES = 3

# Continuous recording. 0.0 saves every frame the camera gives; raise it to
# thin the stream out. Each frame is ~1.2 MB as .npy at 640x480, so a minute
# at full rate is several hundred MB.
RECORD_INTERVAL = 0.0


class IRCameraCapture(QWidget):
    def __init__(self, irCamera: QLabel):
        self.irCameralbl = irCamera

        self.irCameraWorker = irCameraReader()
        self.irCameraWorker.start()
        self.irCameraWorker.ImageUpdate.connect(self.ImageUpdateSlot)

        # Default handlers, so a capture reports something out of the box.
        # Qt allows several listeners per signal, so connect your own alongside
        # these rather than replacing them:
        #   capture.irCameraWorker.BurstFinished.connect(self.my_handler)
        self.irCameraWorker.BurstFinished.connect(self.on_burst)
        self.irCameraWorker.ClosureFinished.connect(self.on_closure)
        self.irCameraWorker.OperatorAlert.connect(self.on_alert)
        self.irCameraWorker.RecordingStopped.connect(self.on_recording_stopped)

    def ImageUpdateSlot(self, Image):
        self.irCameralbl.setPixmap(QPixmap.fromImage(Image))

    # -- default signal handlers; print only, override or add your own -----

    def on_burst(self, position, result, keeper, frames):
        print(f"position {position}: {result.summary()}  "
              f"({len(frames)} frames, kept #{result.keeper})")

    def on_closure(self, pair):
        verdict = "pass" if pair.passed else "FAIL"
        print(f"closure check: IoU {pair.iou:.3f}, shift {pair.shift_px:.2f} px, {verdict}")

    def on_alert(self, position, reason):
        print(f"OPERATOR ALERT - position {position}: {reason}")

    def on_recording_stopped(self, count, folder):
        print(f"recording stopped: {count} frames in {folder}")

    def SaveImages(self, imageFolder: str, imageName: str, arrayName):
        self.irCameraWorker.SaveImages(imageFolder=imageFolder, irName=imageName, arrayName=arrayName)

    def PauseFeed(self, state):
        self.irCameraWorker.PauseFeed(state)

    def StopCamera(self):
        self.irCameraWorker.stop()

    def checkirCameraAttached(self):
        print("checkirCameraAttached")

    # -- motion-gated capture, forwarded to the worker --------------------

    def CapturePosition(self, position):
        """Queue a gated burst at `position`. Returns immediately.

        Call AFTER the rig has been told to move. The verdict arrives on the
        worker's BurstFinished signal, or OperatorAlert if it gives up.
        """
        self.irCameraWorker.CapturePosition(position)

    def CaptureClosure(self):
        """Queue the end-of-session check. Returns immediately."""
        self.irCameraWorker.CaptureClosure()

    def SaveLog(self, path):
        self.irCameraWorker.SaveLog(path)

    def SaveBurst(self, folder, prefix="burst"):
        """Write every frame of the last burst, not just the one that was kept."""
        return self.irCameraWorker.SaveBurst(folder, prefix)

    def StartRecording(self, folder, prefix="rec", **kwargs):
        """Save every frame to `folder` until StopRecording(). Feed keeps running."""
        self.irCameraWorker.StartRecording(folder, prefix, **kwargs)

    def StopRecording(self):
        return self.irCameraWorker.StopRecording()

    def SaveRecordingLog(self, path):
        """Frame-to-frame measurements as CSV. The calibration data set."""
        self.irCameraWorker.SaveRecordingLog(path)


class irCameraReader(QThread):
    ImageUpdate = Signal(QImage)
    BurstFinished = Signal(int, object, object, object)  # position, BurstResult, kept frame, all frames
    ClosureFinished = Signal(object)              # PairResult
    OperatorAlert = Signal(int, str)              # position, why it gave up
    RecordingProgress = Signal(int, object)       # frames saved, PairResult vs the one before (or None)
    RecordingStopped = Signal(int, str)           # frames saved, folder

    this_dir = os.path.abspath(os.path.dirname(__file__))
    dep1 = ct.cdll.LoadLibrary(os.path.join(this_dir, 'opencv_core440.dll'))
    irCamDLL = ct.cdll.LoadLibrary(os.path.join(this_dir, 'icisdk_x64.dll'))

    def __init__(self, thresholds=None):
        super().__init__()
        self.thresholds = thresholds or md.Thresholds()
        self.ThreadActive = False
        self.feedNoShow = False
        self.Temperatures = None    # latest frame, for SaveImages
        self.ir_img = None          # latest display image, for SaveImages
        self.first_capture = None   # reference for the end-of-session check
        self.log = []               # one row per attempt, for threshold tuning
        self.last_burst = []        # every frame of the most recent burst, kept or not
        self.recording_log = []     # one row per recorded frame pair
        self._pending = None        # (kind, arg) picked up by the run loop
        self._recording = None      # settings dict while recording, else None
        self._lock = threading.Lock()

    def MakeTempArray(self, h, w, Ptr):
        """Temperatures in degrees C as an (h, w) array, one value per pixel.

        as_array points at the SDK's own buffer, which the next grab
        overwrites, so the copy is required for any frame that outlives
        this call. Without it, every frame in a burst aliases the same
        memory and the motion check compares a frame against itself.
        """
        return np.ctypeslib.as_array(Ptr, shape=(h, w)).copy()

    def ReadTemperatureFrame(self):
        """One frame of temperatures in degrees C, oriented to match the visibles."""
        self.irCamDLL.GetTemperatureImage.restype = ct.POINTER(ct.c_float)
        TemperatuerPointer = self.irCamDLL.GetTemperatureImage()
        if not TemperatuerPointer:
            return None

        tempArray = self.MakeTempArray(self.ht, self.wd, TemperatuerPointer)
        flipped = cv2.flip(tempArray, 1)  # flip image so it matches visibles
        return cv2.rotate(flipped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def ToDisplayImage(self, temperatures):
        """Jet-colormapped 8-bit image, the same look as before."""
        IR_img = cv2.normalize(temperatures, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        ir_rgb = cv2.cvtColor(IR_img, cv2.COLOR_GRAY2RGB)
        return cv2.applyColorMap(ir_rgb, cv2.COLORMAP_JET)

    def run(self):
        self.ThreadActive = True

        self.irCamDLL.StartCamera(0, 1)

        self.ht = self.irCamDLL.GetImgHeight()
        self.wd = self.irCamDLL.GetImgWidth()

        print("Starting IR Camera", self.ht)

        while self.ThreadActive:

            # A queued capture takes the camera for itself; the feed resumes
            # once it is done.
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is not None:
                kind, arg = pending
                if kind == "position":
                    self._run_position(arg)
                elif kind == "closure":
                    self._run_closure()
                continue

            frame = self.ReadTemperatureFrame()

            if frame is not None:
                self.Temperatures = frame
                self.ir_img = self.ToDisplayImage(frame)

                ConvertToQtFormat = QImage(self.ir_img.data, self.ir_img.shape[1], self.ir_img.shape[0], QImage.Format_BGR888)
                Pic = ConvertToQtFormat.scaled(815, 615, Qt.KeepAspectRatio)

                if not self.feedNoShow:
                    self.ImageUpdate.emit(Pic)

                with self._lock:
                    recording = self._recording
                if recording is not None:
                    self._record_frame(recording, frame)

            else:
                print("NULL!!")

    # -- the motion gate ---------------------------------------------------

    def CapturePosition(self, position):
        """Queue a gated burst. Safe to call from the GUI thread."""
        with self._lock:
            self._pending = ("position", position)

    def CaptureClosure(self):
        """Queue the end-of-session check. Safe to call from the GUI thread."""
        with self._lock:
            self._pending = ("closure", None)

    def GrabBurst(self, count=BURST_FRAMES, seconds=BURST_SECONDS):
        """Capture `count` frames spread over `seconds`, oldest first.

        Runs on the camera thread only. Reading flat out would return the same
        frame more than once whenever the camera is slower than the loop, so
        the burst is paced instead.
        """
        frames = []
        interval = seconds / max(count - 1, 1)
        while len(frames) < count and self.ThreadActive:
            started = time.perf_counter()
            frame = self.ReadTemperatureFrame()
            if frame is not None:
                frames.append(frame)
            remaining = interval - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
        return frames

    def _run_position(self, position, settle=SETTLE_SECONDS, retries=MAX_RETRIES):
        """Settle, burst, check. Retry on motion, then hand off to the operator.

        Every attempt is logged whether it passes or fails - those numbers are
        what the threshold calibration will be built from.
        """
        time.sleep(settle)

        result = None
        for attempt in range(1, retries + 1):
            frames = self.GrabBurst()
            if len(frames) < 2:
                break

            result = md.check_burst(frames, self.thresholds)
            self.last_burst = frames

            self.log.append({
                "position": position,
                "attempt": attempt,
                "iou": result.worst_iou,
                "shift_px": result.worst_shift,
                "passed": result.passed,
            })

            if result.passed:
                keeper = frames[result.keeper]
                if self.first_capture is None:
                    self.first_capture = keeper
                self.BurstFinished.emit(position, result, keeper, frames)
                return

            print(f"position {position} attempt {attempt}: {result.summary()}")

        reason = "no frames from the camera"
        if result is not None and result.reasons:
            reason = result.reasons[0]
        self.OperatorAlert.emit(position, f"{retries} attempts failed - {reason}")

    def _run_closure(self, settle=SETTLE_SECONDS):
        """End-of-session check. Queue it after rotating back to the first angle.

        Catches the patient resettling BETWEEN positions, which the per-position
        checks cannot see. Assumes the rig returns to the same spot repeatably -
        verify that with an empty rig first.
        """
        if self.first_capture is None:
            self.OperatorAlert.emit(-1, "no first capture to compare against")
            return

        time.sleep(settle)
        frames = self.GrabBurst()
        if not frames:
            self.OperatorAlert.emit(-1, "no frames from the camera")
            return

        self.ClosureFinished.emit(
            md.check_closure(self.first_capture, frames[0], self.thresholds))

    def SaveBurst(self, folder, prefix="burst"):
        """Write every frame of the most recent burst, not just the keeper.

        Each frame goes down twice: a .npy of the raw temperatures and a .jpg
        for looking at. Returns how many frames were written.
        """
        os.makedirs(folder, exist_ok=True)
        for i, frame in enumerate(self.last_burst):
            base = os.path.join(folder, f"{prefix}_{i:02d}")
            np.save(base + ".npy", frame)
            cv2.imwrite(base + ".jpg", self.ToDisplayImage(frame))
        return len(self.last_burst)

    # -- continuous recording ----------------------------------------------

    def StartRecording(self, folder, prefix="rec", interval=RECORD_INTERVAL,
                       save_jpg=True, log_metrics=True, max_frames=None):
        """Save every frame to `folder` until StopRecording().

        The live feed keeps running. Safe to call from the GUI thread.

        interval     seconds between saved frames; 0.0 saves all of them
        save_jpg     also write a viewable .jpg beside each .npy
        log_metrics  measure each frame against the one before and log it.
                     This is how the calibration run gets its numbers.
        max_frames   stop automatically after this many
        """
        os.makedirs(folder, exist_ok=True)
        self.recording_log = []
        with self._lock:
            self._recording = {
                "folder": folder, "prefix": prefix, "interval": interval,
                "save_jpg": save_jpg, "log_metrics": log_metrics,
                "max_frames": max_frames, "count": 0,
                "previous": None, "last_at": 0.0,
            }

    def StopRecording(self):
        """Stop a recording. Safe to call from the GUI thread, or when idle."""
        with self._lock:
            recording, self._recording = self._recording, None
        if recording is not None:
            self.RecordingStopped.emit(recording["count"], recording["folder"])
        return 0 if recording is None else recording["count"]

    def _record_frame(self, rec, frame):
        """Save one frame. Camera thread only."""
        now = time.perf_counter()
        if rec["interval"] > 0 and (now - rec["last_at"]) < rec["interval"]:
            return
        rec["last_at"] = now

        base = os.path.join(rec["folder"], f"{rec['prefix']}_{rec['count']:05d}")
        np.save(base + ".npy", frame)
        if rec["save_jpg"]:
            cv2.imwrite(base + ".jpg", self.ToDisplayImage(frame))

        pair = None
        if rec["log_metrics"] and rec["previous"] is not None:
            pair = md.compare(rec["previous"], frame, thresholds=self.thresholds)
            self.recording_log.append({
                "frame": rec["count"],
                "iou": pair.iou,
                "shift_px": pair.shift_px,
                "dx": pair.dx,
                "dy": pair.dy,
                "passed": pair.passed,
            })

        rec["previous"] = frame
        rec["count"] += 1
        self.RecordingProgress.emit(rec["count"], pair)

        if rec["max_frames"] and rec["count"] >= rec["max_frames"]:
            self.StopRecording()

    def SaveRecordingLog(self, path):
        """Frame-to-frame measurements as CSV. Plot this to set the thresholds."""
        with open(path, "w") as handle:
            handle.write("frame,iou,shift_px,dx,dy,passed\n")
            for row in self.recording_log:
                handle.write("{frame},{iou:.4f},{shift_px:.3f},"
                             "{dx:.3f},{dy:.3f},{passed}\n".format(**row))

    def SaveLog(self, path):
        """Dump every attempt as CSV. This is the calibration data set."""
        with open(path, "w") as handle:
            handle.write("position,attempt,iou,shift_px,passed\n")
            for row in self.log:
                handle.write("{position},{attempt},{iou:.4f},"
                             "{shift_px:.3f},{passed}\n".format(**row))

    # -- housekeeping ------------------------------------------------------

    def PauseFeed(self, state):
        self.feedNoShow = state

    # Old spelling, kept so existing callers keep working.
    pauseFeed = PauseFeed

    def SaveImages(self, imageFolder, irName, arrayName):
        print('Saving image:'+ imageFolder + irName)
        cv2.imwrite(imageFolder + irName, self.ir_img)
        np.savetxt(imageFolder + arrayName, self.Temperatures, delimiter=",")

    def stop(self):
        self.ThreadActive = False
        print("Turning off IR Camera")
        self.irCamDLL.StopCamera()

        self.quit()
