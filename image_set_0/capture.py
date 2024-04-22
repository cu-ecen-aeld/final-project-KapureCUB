import numpy as np
import cv2 as cv
import glob

chessboardSize = (7,5)
frameSize = (1440,1080)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1].T.reshape(-1,2)]

objPoints = []
imgPoints = []

images = glob.glob('*.jpg')

for image in images:
    print(image)(
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        objPoints.append(objp)
        corners2 = cv.cornerSubpix(gray, corners, (11,11), (-1,-1), criteria)
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


## generating undistored image

img = cv.imread('img5.jpg')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

### only undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('img5result.jpg', dst)

### undistort with remapping
mapx, mapy = cvinitUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('img5resultRemap.jpg', dst)

## error calculation
mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(objPoints)
    mean_error += error

print("\ntotalerror: {}\n".format(mean_error/len(objPoints)))
