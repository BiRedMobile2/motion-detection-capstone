# Class to interact with the Oak-D camera, sending images to UI and saving to folder.
# Adapted from Carlos' code
# Scott N. 9/26/2022

from ast import Try
from itertools import count
import os
import cv2
import depthai as dai
from PySide6.QtCore import *
from PySide6.QtWidgets import QWidget, QLabel
from  PySide6.QtGui import QImage, QPixmap


pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

rgbOut = pipeline.create(dai.node.XLinkOut)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.initialControl.setManualFocus(80) # smaller further focus - 20 far field clear, 80 pretty good, center and CL+3"; 100 CL fuzzy, CL+3 not clear.; 70 CL+ not clear

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

# Linking
camRgb.isp.link(rgbOut.input)
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)

class visCameraCapture(QWidget):
    def __init__(self, visCamera: QLabel):
        self.visCameralbl = visCamera

        self.visCameraWorker = visCameraReader()
        self.visCameraWorker.start()
        self.visCameraWorker.ImageUpdate.connect(self.ImageUpdateSlot)

    def ImageUpdateSlot(self, Image):
        self.visCameralbl.setPixmap(QPixmap.fromImage(Image))

    def SaveImages(self, imageFolder: str, visName: str, stRfileName, stLfileName):
        self.visCameraWorker.SaveImages(imageFolder=imageFolder, visName=visName, stRfileName=stRfileName, stLfileName=stLfileName)

    def PauseFeed(self, state):
        self.visCameraWorker.pauseFeed(state)

    def checkVisCameraAttached(self):
        print("checkVisCameraAttached")
        cams = dai.getConnectedCameras()
        print("cams:", cams)
        if count(cams) != 3:
            cameraError = "Issue communicating with Oak-D stero camera."
            print(cameraError)


class visCameraReader(QThread):
    ImageUpdate = Signal(QImage)
    def run(self):
        self.ThreadActive = True

        script_dir = os.path.dirname(__file__) 
        rel_path = "NoFeedVisCamera.png"
        placeHolderImage = os.path.join(script_dir, rel_path)

 
        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            self.frameRgb = None
            self.frameleft = None
            self.frameright = None

            self.feedNoShow = False

            while self.ThreadActive:
                latestPacket = {}
                latestPacket["rgb"] = None
                latestPacket["left"] = None
                latestPacket["right"] = None

                queueEvents = device.getQueueEvents(("rgb", "left", "right" ))

                for queueName in queueEvents:
                    packets = device.getOutputQueue(queueName).tryGetAll()
                    if len(packets) > 0:
                        latestPacket[queueName] = packets[-1]

                if latestPacket["rgb"] is not None:   #not self.feedNoShow and 
                    self.frameRgb = latestPacket["rgb"].getCvFrame()
                    Image = cv2.cvtColor(self.frameRgb, cv2.COLOR_BGR2RGB)
                    FlippedImage = cv2.flip(Image, 1)
                    ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                    Pic = ConvertToQtFormat.scaled(815, 615, Qt.KeepAspectRatio)
                    self.ImageUpdate.emit(Pic)

                # elif(self.feedNoShow):
                #     pic = QImage(placeHolderImage)
                #     self.ImageUpdate.emit(pic)

                if latestPacket["left"] is not None:
                    self.frameleft = latestPacket["left"].getFrame()


                if latestPacket["right"] is not None:
                    self.frameright = latestPacket["right"].getFrame()

            
    def pauseFeed(self, state):
        self.feedNoShow = state

    def SaveImages(self, imageFolder, visName, stRfileName, stLfileName):
        print('Saving image:'+ imageFolder + visName)
        cv2.imwrite(imageFolder + visName, self.frameRgb)
        cv2.imwrite(imageFolder + stRfileName, self.frameright)        
        cv2.imwrite(imageFolder + stLfileName, self.frameleft)

    def stop(self):
        self.ThreadActive = False
        self.quit()       
