import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt


chessboardSize = (8,6)
#frameSize = (1440,1080)
frameSize = (1280,720)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

objPoints = []
imgPoints = []

images = glob.glob('*.jpg')

for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        print("corners found in image\n")
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)


cv.destroyAllWindows()


#### calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

print("Camera calibrated: ", ret)
print("\nCamera Matrix:\n", cameraMatrix)
print("\nDistorsion Parameters:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)

still_list = ["still1_L.jpg", "still2_R.jpg"]
#still_list = ["nithar1_L.jpg", "nithar2_R.jpg"]

for img in still_list:
    ## generating undistored image
    img_name = img
    print("performing on {}\n".format(img))
    img = cv.imread(img)
    h, w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

    ### only undistort
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    #cv.imwrite('img1result.jpg', dst)

    ### undistort with remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    string_name = "remap_" + img_name
    print(string_name)
    print("\n")
    cv.imwrite(string_name, dst)


## error calculation
mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(objPoints)
    mean_error += error

print("\ntotalerror: {}\n".format(mean_error/len(objPoints)))


## depth map
left_image = cv.imread('remap_{}'.format(still_list[0]), cv.IMREAD_GRAYSCALE)
right_image = cv.imread('remap_{}'.format(still_list[1]), cv.IMREAD_GRAYSCALE)


stereo = cv.StereoBM_create(numDisparities=80, blockSize=9)
# For each pixel algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity map
depth = stereo.compute(left_image, right_image)

print(depth)

cv.imshow("Left", left_image)
cv.imshow("right", right_image)

plt.imshow(depth)
plt.axis('off')
plt.show()