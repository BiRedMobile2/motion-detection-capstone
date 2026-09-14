import ctypes as ct
import os
import struct
import numpy
import cv2

this_dir = os.path.abspath(os.path.dirname(__file__))

dep1 = ct.cdll.LoadLibrary(os.path.join(this_dir, 'opencv_core440.dll'))
irCamDLL = ct.cdll.LoadLibrary(os.path.join(this_dir, 'icisdk_x64.dll'))

# From SDKTestDlg.cpp
# void
# CSDKTestDlg::FetchAndDisplayFrame()
# {
# 	auto irImage = IRImage {};
# 	irImage.width = m_ImgWidth;
# 	irImage.height = m_ImgHeight;
# 	irImage.numberOfChannels = 3;
# 	irImage.bitsPerChannel = 8;
# 	auto* rawIRImageData = m_SDKInterface.GetTemperatureImage();
# 	if ( nullptr != rawIRImageData )
# 	{
# 		auto irImageData = std::vector <float>( m_ImgWidth * m_ImgHeight );
# 		memcpy( irImageData.data() , rawIRImageData , m_ImgWidth * m_ImgHeight * sizeof( decltype( irImageData.at( 0 ) ) ) );
# 		irImage.imageData = GetDisplayImage( irImageData );
# 		m_ImgDisplayWnd.Display( irImage );
# 		DisplayCenterPointTemperature( irImageData );
# 	}
# }



irCamDLL.StartCamera(0,1)

ir_path = True

while ir_path == True:
    ht = irCamDLL.GetImgHeight()
    wd = irCamDLL.GetImgWidth()
    img = irCamDLL.GetRawImage()

    print(irCamDLL.GetCameraSerialNumber())

    countTries = 0

    ##########################
    imagePointer = irCamDLL.GetTemperatureImage()

    print("imagePointer:", imagePointer)
    if imagePointer:

        ip = ct.cast(imagePointer, ct.POINTER(ct.c_float))
        
        # print("Try memmove")

        # buf = (ct.c_float * ht * wd)()
        # ct.memmove(buf, ip, ct.sizeof(buf))

        # print("buf cell:", buf[3])

        # print("end memmove")




        # a = numpy.ctypeslib.as_array(ct.POINTER(ct.c_float).from_address(imagePointer), shape=(ht, wd))
        A = numpy.ctypeslib.as_array(ip, shape=(ht, wd ))
        print("past as array")
        # print("Type:", type(a))
        # print("Size:", a.size)


        #type of array
        print("Type:", type(A))
        #Shape of array
        print("Shape:", A.shape)
        #no. of dimensions
        print("Rank:", A.ndim)
        #size of array
        print("Size:", A.size)
        #type of each element in the array
        print("Element type:", A.dtype)
        tmp = A[:2, :2]
        print("value:\n", tmp)
        print("done")

        ir_img = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        ir_rgb = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)
        ir_colormap = cv2.applyColorMap(ir_rgb, cv2.COLORMAP_JET)
        cv2.imshow("IR Camera", ir_colormap)
    else:
        print("NULL")


key = cv2.waitKey(1)
if key == ord('q'):
    ir_path = False


irCamDLL.StopCamera()
cv2.destroyAllWindows()


