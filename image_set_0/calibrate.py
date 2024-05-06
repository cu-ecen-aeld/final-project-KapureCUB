'''
This script contains the code to calculate the camera calibration parameters of a particular image set provided.
The scripts uses the chessboard corner method to calculate the camera matrix and distortion parameters.

Please refer the instructions mentioned here -
https://github.com/cu-ecen-aeld/buildroot-assignments-base/wiki/OpenCV-Stereo-Vision

Additional reference used for this script -
> https://github.com/niconielsen32
> https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
> https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
'''

import numpy as np
import cv2 as cv
import glob

# The size of the chessboard pattern displayed in the sample image.
# The tuple contains the number of squares on the horizontal and vertical axes respectively
# Modify this field to match with your calibration image set
chessboardSize = (8,6)

# Frame size parameter.
# The tuple below signifies the image resolution of the captures images on which the calibration is performed.
# Edit the below numbers to match your calibration image set
frameSize = (1440,1080)

# Condition to exit the iteration for corner subpixel calculation
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

# Defining object and image points array
objPoints = []
imgPoints = []

# The image set used with .ppm extension.
# The code is compatible with image set of .png and .jpg extensions as well
# The images reside in the current working directory
images = glob.glob('*.ppm')

for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # calling the function to calulate corners in the given pattern
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        print("Corners found in image\n")
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()


# Calibration parameters
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

print("Camera calibrated: ", ret)
print("\nCamera Matrix:\n", cameraMatrix)
print("\nDistorsion Parameters:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)
