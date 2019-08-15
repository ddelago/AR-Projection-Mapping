# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle
import sys

# Check for camera calibration data
if not os.path.exists('./calibration/ProCamCalibration.pckl'):
    print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
    exit()
else:
    f = open('./calibration/ProCamCalibration.pckl', 'rb')
    (_, _, _, _, cameraMatrix, distCoeffs, _) = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove ./calibration/ProCamCalibration.pckl and recalibrate your camera with CalibrateCamera.py.")
        exit()

def onClick(event, x, y, flags, params):
    global corner
    # left-click event value is 2
    if event == 1:
        # store the coordinates of the click event
        cornersTransform[corner] = [x,y]
        corner += 1

def four_point_transform(image, cornersTransform):

	(tl, tr, br, bl) = cornersTransform
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(cornersTransform, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# Create grid board object we're using in our stream
board = aruco.GridBoard_create(
        markersX=1,
        markersY=1,
        markerLength=0.09,
        markerSeparation=0.01,
        dictionary=ARUCO_DICT)

cornersTransform = np.zeros((4, 2), dtype="float32")
corner = 0

# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None
axis = np.float32([[-.5,-.5,0], [-.5,.5,0], [.5,.5,0], [.5,-.5,0],
                   [-.5,-.5,1],[-.5,.5,1],[.5,.5,1],[.5,-.5,1] ])

aruco1 = cv2.imread('./patterns/test_marker1.jpg')
aruco2 = cv2.imread('./patterns/test_marker2.jpg')
aruco3 = cv2.imread('./patterns/test_marker3.jpg')
aruco4 = cv2.imread('./patterns/test_marker4.jpg')

transformed = False
height = 720
width = 1280

# Make output image fullscreen
cv2.namedWindow('ProjectImage',cv2.WINDOW_NORMAL)
cv2.namedWindow('InputImage',cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('ProjectImage', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam.set(3, 1280)
cam.set(4, 720)
while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, ProjectImage = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(ProjectImage, cv2.COLOR_BGR2GRAY)

        # Display our image
        cv2.imshow('InputImage', ProjectImage)
        cv2.setMouseCallback('InputImage', onClick)

        # Make background black
        ProjectImage = np.zeros((720, 1280, 3), np.uint8)
        ProjectImage[:] = (255, 255, 255)
        ProjectImage[ 0:0+aruco1.shape[0], 0:0+aruco1.shape[1]] = aruco1
        ProjectImage[ 0:0+aruco1.shape[0], 0-aruco1.shape[1]:width] = aruco2
        ProjectImage[ 0-aruco1.shape[0]:height, 0:0+aruco1.shape[1]] = aruco3
        ProjectImage[ 0-aruco1.shape[0]:height, 0-aruco1.shape[1]:width] = aruco4
        
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
  
        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image = gray,
                board = board,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = cameraMatrix,
                distCoeffs = distCoeffs)   

        # Outline all of the markers detected in our image
        # ProjectImage = aruco.drawDetectedMarkers(ProjectImage, corners, ids, borderColor=(0, 0, 255))
        ProjectImage = aruco.drawDetectedMarkers(ProjectImage, corners, borderColor=(0, 0, 255))
        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
    
        # Aruco marker found
        imgptsAll = []
        if ids is not None and len(ids) > 0:
            # Estimate the posture per each Aruco marker
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)

            for rvec, tvec in zip(rvecs, tvecs):
                # project 3D points to image plane
                try:
                    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs)
                    imgptsAll.append(imgpts)
                except:
                    continue
                
                if len(sys.argv) == 2 and sys.argv[1] == 'cube':
                    ProjectImage = drawCube(ProjectImage, corners, imgpts)    
                # ProjectImage = aruco.drawAxis(ProjectImage, cameraMatrix, distCoeffs, rvec, tvec, 1)
        
            # If four corners found, transform image
            topLeft = [1]
            topRight = [2]
            botLeft = [3]
            botRight = [4]
            if (topLeft in ids) and (topRight in ids) and (botLeft in ids) and (botRight in ids) and not transformed:
                print(imgptsAll[np.where(ids == topLeft)[0][0]])
                print(imgptsAll[np.where(ids == topLeft)[0][0]][0][0])
                cornersTransform[0] = imgptsAll[np.where(ids == topLeft)[0][0]][1][0]
                cornersTransform[1] = imgptsAll[np.where(ids == topRight)[0][0]][2][0]
                cornersTransform[2] = imgptsAll[np.where(ids == botRight)[0][0]][3][0]
                cornersTransform[3] = imgptsAll[np.where(ids == botLeft)[0][0]][0][0]
                # print(cornersTransform)
                ProjectImage = four_point_transform(ProjectImage, cornersTransform)
                transformed = True
            
            if transformed:
                ProjectImage = four_point_transform(ProjectImage, cornersTransform)

        ProjectImage[ 0:0+aruco1.shape[0], 0:0+aruco1.shape[1]] = aruco1
        ProjectImage[ 0:0+aruco1.shape[0], 0-aruco1.shape[1]:width] = aruco2
        ProjectImage[ 0-aruco1.shape[0]:height, 0:0+aruco1.shape[1]] = aruco3
        ProjectImage[ 0-aruco1.shape[0]:height, 0-aruco1.shape[1]:width] = aruco4

        cv2.imshow('ProjectImage', ProjectImage)

    # Reset corner calibration
    if cv2.waitKey(1) == ord('r'):
        print("Corners reset")
        cornersTransform = np.zeros((4, 2), dtype="float32")
        corner = 0
    
    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()