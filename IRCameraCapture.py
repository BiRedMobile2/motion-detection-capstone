# Class to interact with the Oak-D camera, sending images to UI and saving to folder.
# Adapted from Carlos' code
# Scott N. 9/26/2022

import os
import cv2
import ctypes as ct
import numpy as np
from PySide6.QtCore import *
from PySide6.QtWidgets import QWidget, QLabel
from  PySide6.QtGui import QImage, QPixmap



class IRCameraCapture(QWidget):
    def __init__(self, irCamera: QLabel):
        self.irCameralbl = irCamera

        self.irCameraWorker = irCameraReader()
        self.irCameraWorker.start()
        self.irCameraWorker.ImageUpdate.connect(self.ImageUpdateSlot)

    def ImageUpdateSlot(self, Image):
        self.irCameralbl.setPixmap(QPixmap.fromImage(Image))

    def SaveImages(self, imageFolder: str, imageName: str, arrayName):
        self.irCameraWorker.SaveImages(imageFolder=imageFolder, irName=imageName, arrayName=arrayName)

    def PauseFeed(self, state):
        self.irCameraWorker.pauseFeed(state)

    def StopCamera(self):
        self.irCameraWorker.stop()

    def checkirCameraAttached(self):
        print("checkirCameraAttached")
    

class irCameraReader(QThread):
    ImageUpdate = Signal(QImage)

    this_dir = os.path.abspath(os.path.dirname(__file__))
    dep1 = ct.cdll.LoadLibrary(os.path.join(this_dir, 'opencv_core440.dll'))
    irCamDLL = ct.cdll.LoadLibrary(os.path.join(this_dir, 'icisdk_x64.dll'))


    def MakeTempArray(self,h,w,Ptr):
        Temp_vec = np.zeros((h*w,1),dtype=float)

        for i in range(h*w):
            Temp_vec[i] = Ptr[i]

        Temp_Array = Temp_vec.reshape((h,w))
        
        '''for row  in range(h):
            for col in range(w):
                Temp_Array[row,col] = Ptr[row][col]'''

        return Temp_Array

    def run(self):
        self.ThreadActive = True

        self.irCamDLL.StartCamera(0,1)

        ht = self.irCamDLL.GetImgHeight()
        wd = self.irCamDLL.GetImgWidth()

        # script_dir = os.path.dirname(__file__) 
        # rel_path = "NoFeedirCamera.png"
        # placeHolderImage = os.path.join(script_dir, rel_path)

        print("Starting IR Camera", ht)
 
        while self.ThreadActive:

            self.irCamDLL.GetTemperatureImage.restype = ct.POINTER(ct.c_float)
            TemperatuerPointer = self.irCamDLL.GetTemperatureImage()

            if TemperatuerPointer:
                tempArray = self.MakeTempArray(ht,wd,TemperatuerPointer)

                self.Temperatures = cv2.flip(tempArray,1) # flip image so it matches visibles
                self.Temperatures = cv2.rotate(self.Temperatures, cv2.ROTATE_90_COUNTERCLOCKWISE)

                IR_img = cv2.normalize(self.Temperatures,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
                ir_rgb = cv2.cvtColor(IR_img, cv2.COLOR_GRAY2RGB)
                self.ir_img = cv2.applyColorMap(ir_rgb, cv2.COLORMAP_JET)

                ConvertToQtFormat = QImage(self.ir_img.data, self.ir_img.shape[1], self.ir_img.shape[0], QImage.Format_BGR888)
                Pic = ConvertToQtFormat.scaled(815, 615, Qt.KeepAspectRatio)

                # if latestPacket["rgb"] is not None:   #not self.feedNoShow and 
                self.ImageUpdate.emit(Pic)

                # elif(self.feedNoShow):
                #     pic = QImage(placeHolderImage)
                #     self.ImageUpdate.emit(pic)

            else:
                print("NULL!!")
            
    def pauseFeed(self, state):
        self.feedNoShow = state

    def SaveImages(self, imageFolder, irName, arrayName):
        print('Saving image:'+ imageFolder + irName)
        cv2.imwrite(imageFolder + irName, self.ir_img)
        np.savetxt(imageFolder + arrayName, self.Temperatures, delimiter=",")
 
    def stop(self):
        self.ThreadActive = False
        print("Turning off IR Camera")
        self.irCamDLL.StopCamera()

        self.quit()       
