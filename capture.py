import numpy as np
import cv2 as cv
import os
import socket
import time
import signal
import sys

######## CAMERA PARAMETERS ############
cameraMatrix = np.array([[1.24658345e+03, 0.00000000e+00, 6.44698034e+02],
                         [0.00000000e+00, 1.24438489e+03, 3.50116318e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distortionPara = np.array([0.15825909, -0.18346304, -0.03790805, -0.03332055, 0.04612853])
stereoParameters = [80, 9]  # [numDisparities, blocksize]
still_list = ["still1_L.ppm", "still2_R.ppm"]
#######################################

######## DRIVER PARAMETERS ############
captureLeftCmd = "capture /dev/video2"
captureRightCmd = "capture /dev/video0"
copyFrame30Cmd = "cp fram00000030.ppm "
renameStillLeft = "still1_L.ppm"
renameStillRight = "still2_R.ppm"
defaultDepthImage = 'depth.jpg'
#######################################

######## SOCKET PARAMETERS ############
# configure these for socket connection based on server
serverAddress = '10.0.0.46'
serverPort = 1002
imageChunkToSend = 2048
#######################################


########## SIGNAL HANDLERS ############
def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(-1)


signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)
#######################################

# create a socket for transferring images to server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # AF_INET = IP, SOCK_STREAM = TCP

# connect to server
print("Connecting to server..")
client.connect((serverAddress, serverPort))


class StereoVisionPi:
    def __init__(self, images):
        self.img_list = images
        self.img_str_L = self.img_list[0]
        self.img_str_R = self.img_list[1]
        self.img_str = ""
        self.height = ""
        self.width = ""
        self.newCameraMatrix = ""
        self.roi = ""
        self.mapx = ""
        self.mapy = ""
        self.distortion = ""
        self.img = ""
        self.abscissa = ""
        self.ordinate = ""
        self.leftImage = ""
        self.rightImage = ""
        self.stereo = ""
        self.depth = ""
        self.heatmap = ""

    def depth_map(self, leftimg, rightimg):
        self.leftImage = cv.imread(leftimg, cv.IMREAD_GRAYSCALE)
        self.rightImage = cv.imread(rightimg, cv.IMREAD_GRAYSCALE)
        self.stereo = cv.StereoBM_create(numDisparities=stereoParameters[0], blockSize=stereoParameters[1])
        self.depth = self.stereo.compute(self.leftImage, self.rightImage)
        self.heatmap = cv.applyColorMap(self.depth, cv.COLORMAP_HOT)

        return self.heatmap

    def perform_depth_mapping(self):
        for self.img in self.img_list:
            self.img_str = self.img
            print("Performing rectification on {}".format(self.img_str))
            self.img = cv.imread(self.img)
            self.height, self.width = self.img.shape[:2]
            self.newCameraMatrix, self.roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distortionPara,
                                                                          (self.width, self.height),
                                                                          1, (self.width, self.height))
            # un-distortion with remapping
            self.mapx, self.mapy = cv.initUndistortRectifyMap(cameraMatrix, distortionPara, None, self.newCameraMatrix,
                                                              (self.width, self.height), 5)
            self.distortion = cv.remap(self.img, self.mapx, self.mapy, cv.INTER_LINEAR)
            self.abscissa, self.ordinate, self.width, self.height = self.roi
            self.distortion = self.distortion[self.ordinate:self.ordinate + self.height,
                              self.abscissa:self.abscissa + self.width]
            cv.imwrite(("remap_" + self.img_str), self.distortion)

        return self.depth_map(("remap_" + self.img_str_L), ("remap_" + self.img_str_R))


imgObj = StereoVisionPi(still_list)

#while True:
try:
    # call the capture command on camera R
    ret = os.system(captureLeftCmd)
    if ret == 0:
        ret = os.system(copyFrame30Cmd + renameStillLeft)
        if ret == 0:
            # copy frame30 as R.img
            ret = os.system(captureRightCmd)
            if ret == 0:
                ret = os.system(copyFrame30Cmd + renameStillRight)
                if ret == 0:
                    # perform depth estimation on rectified images
                    depthMap = imgObj.perform_depth_mapping()
                    cv.imwrite("depth.jpg", depthMap)
                    print("Successfully created depth map")

                    # send data via socket
                    file = open(defaultDepthImage, 'rb')   # open in read binary mode
                    image_data = file.read(imageChunkToSend)
                    while image_data:
                        try:
                            client.send(image_data)
                        except:
                            print("Error sending image to server.")
                            break
                        image_data = file.read(imageChunkToSend)
                    print("Image sent to server")
                    file.close()
                    time.sleep(1)
                else:
                    print("Error in saving right image")
            else:
                print("Error in calling capture on right camera")
        else:
            print("Error in saving left image")
    else:
        print("Error in calling capture on left camera")
except:
    client.close()
    print("Error in computing depth capture")
    #break

finally:
    client.close()
    print("Keyboard interrupt. Exiting..")
    #break


