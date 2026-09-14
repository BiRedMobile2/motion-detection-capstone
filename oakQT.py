import sys
import cv2
import depthai as dai
from  PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *


pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)

rgbOut = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.initialControl.setManualFocus(130)

# Linking
camRgb.isp.link(rgbOut.input)

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()


class Worker1(QThread):
    ImageUpdate = Signal(QImage)
    def run(self):
        self.ThreadActive = True
 
        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            frameRgb = None
            
            # Window names and sizes
            # cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)

            # cv2.resizeWindow('RGB', 1920,1080)
    
            while self.ThreadActive:
                latestPacket = {}
                latestPacket["rgb"] = None
            
                queueEvents = device.getQueueEvents(("rgb" ))

                for queueName in queueEvents:
                    packets = device.getOutputQueue(queueName).tryGetAll()
                    if len(packets) > 0:
                        latestPacket[queueName] = packets[-1]

                if latestPacket["rgb"] is not None:
                    frameRgb = latestPacket["rgb"].getCvFrame()
                    Image = cv2.cvtColor(frameRgb, cv2.COLOR_BGR2RGB)
                    FlippedImage = cv2.flip(Image, 1)
                    ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                    Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.ImageUpdate.emit(Pic)
    
    
    def stop(self):
        self.ThreadActive = False
        self.quit()



if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())