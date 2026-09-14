import cv2
import depthai as dai

main_path = 'C:/Users/CarlosGutierrez/OneDrive - biredimaging.com/Documents/Software/Image Processing Module/Theoretical Validation/'

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
camRgb.initialControl.setManualFocus(130)

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)


monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Linking
camRgb.isp.link(rgbOut.input)
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)

i=0

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    frameRgb = None
    frameleft = None
    frameright = None
    
    # Window names and sizes
    cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Right', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('RGB', 1920,1080)
    cv2.resizeWindow('Left', 1080,720)
    cv2.resizeWindow('Right', 1080,720)

    while True:
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["left"] = None
        latestPacket["right"] = None
        

        queueEvents = device.getQueueEvents(("rgb", "left", "right"))

        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()
            cv2.imshow('RGB', frameRgb)

        if latestPacket["left"] is not None:
            frameleft = latestPacket["left"].getFrame()
            cv2.imshow('Left', frameleft)

        if latestPacket["right"] is not None:
            frameright = latestPacket["right"].getFrame()
            cv2.imshow('Right', frameright)


        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            if i<10:
                name = '00' + str(i)
            else:
                name = '0' + str(i)
            
            print('Saving image'+ str(i+1))
            cv2.imwrite('Test Images/Test48/RGB/' + name + '.jpg', frameRgb)

            print('Saving Left'+ str(i+1))
            cv2.imwrite('Test Images/Test48/Left/' + name + '.jpg', frameleft)

            print('Saving Right'+ str(i+1))
            cv2.imwrite('Test Images/Test48/Right/' + name + '.jpg', frameright)

            i=i+1