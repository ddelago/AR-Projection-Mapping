# See Calibration method here:
# https://www.morethantechnical.com/2017/11/17/projector-camera-calibration-the-easy-way/

import numpy as np
import cv2
from cv2 import aruco
import pickle
import glob
import os

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

gray = None

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
imgpointsProj = [] # Reprojected points.

# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 8
CHARUCOBOARD_COLCOUNT = 6 
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_50)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard_create(
		squaresX=CHARUCOBOARD_COLCOUNT,
		squaresY=CHARUCOBOARD_ROWCOUNT,
		squareLength=0.04,
		markerLength=0.02,
		dictionary=ARUCO_DICT)

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()

# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
image_size = None # Determined at runtime

cameraMatrix, distCoeffs, rvecs, tvecs = None, None, None, None

# Check for camera calibration data
if not os.path.exists('./calibration/MarkerCalibration.pckl'):
    # images = glob.glob('./pictures/*.jpg')
	cap = cv2.VideoCapture('ChAruco_Circles.webm')
	validCaptures = 0

	# Loop through images glob'ed
	while cap.isOpened():
		# Open the image
		# img = cv2.imread(iname)

		ret, img = cap.read()
		if ret is False:
			break

		# Grayscale the image
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find aruco markers in the query image
		corners, ids, _ = aruco.detectMarkers(
				image=gray,
				dictionary=ARUCO_DICT)
		
		if ids is None:
			continue

		# Outline the aruco markers found in our query image
		img = aruco.drawDetectedMarkers(
				image=img, 
				corners=corners)

		# Get charuco corners and ids from detected aruco markers
		response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
				markerCorners=corners,
				markerIds=ids,
				image=gray,
				board=CHARUCO_BOARD)

		# If a Charuco board was found, let's collect image/corner points
		# Requiring at least 20 squares
		if response > 20:
			# Add these corners and ids to our calibration arrays
			corners_all.append(charuco_corners)
			ids_all.append(charuco_ids)
			
			# Draw the Charuco board we've detected to show our calibrator the board was properly detected
			img = aruco.drawDetectedCornersCharuco(
					image=img,
					charucoCorners=charuco_corners,
					charucoIds=charuco_ids)
		
			# If our image size is unknown, set it now
			if not image_size:
				image_size = gray.shape[::-1]
			
			# Reproportion the image, maxing width or height at 1000
			proportion = max(img.shape) / 1000.0
			img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
			# Pause to display each image, waiting for key press
			cv2.imshow('Charuco board', img)
			if cv2.waitKey(0) == ord('q'):
				break

			validCaptures += 1
			if validCaptures == 40:
				print("40 captures")
				break

	# Destroy any open CV windows
	cv2.destroyAllWindows()

	# Now that we've seen all of our images, perform the camera calibration
	# based on the set of points we've discovered
	calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
			charucoCorners=corners_all,
			charucoIds=ids_all,
			board=CHARUCO_BOARD,
			imageSize=image_size,
			cameraMatrix=None,
			distCoeffs=None)
			
	# Print matrix and distortion coefficient to the console
	print(cameraMatrix)
	print(distCoeffs)
			
	# Save values to be used where matrix+dist is required, for instance for posture estimation
	# I save files in a pickle file, but you can use yaml or whatever works for you
	f = open('./calibration/MarkerCalibration.pckl', 'wb')
	pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
	f.close()
			
	# Print to console our success
	print('Calibration successful. Calibration file used: {}'.format('MarkerCalibration.pckl'))
else:
	print('Camera calibration found, continueing to projector calibration.')
	f = open('./calibration/MarkerCalibration.pckl', 'rb')
	(cameraMatrix, distCoeffs, rvecs, tvecs) = pickle.load(f)
	f.close()
	if cameraMatrix is None or distCoeffs is None:
		print("Calibration issue. Remove ./MarkerCalibration.pckl and recalibrate your camera with CalibrateCamera.py.")
		exit()

# Ray-Plane Intersection Algorithm
def intersectCirclesRaysToBoard(circles, rvec, t, cameraMatrix, dist_coef):
	circles_normalized = cv2.convertPointsToHomogeneous(cv2.undistortPoints(circles, cameraMatrix, dist_coef))
 
	if len(rvec) == 0:
		return None

	rvec = np.array(rvec)
	t = np.array(t)

	R, _ = cv2.Rodrigues(rvec)
 
	# https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
 
	plane_normal = R[2,:] # last row of plane rotation matrix is normal to plane
	plane_point = t.T	 # t is a point on the plane
 
	epsilon = 1e-06
 
	circles_3d = np.zeros((0,3), dtype=np.float32)
 
	for p in circles_normalized:
		ray_direction = p / np.linalg.norm(p)
		ray_point = p
 
		ndotu = plane_normal.dot(ray_direction.T)
 
		if abs(ndotu) < epsilon:
			print ("no intersection or line is within plane")
 
		w = ray_point - plane_point
		si = -plane_normal.dot(w.T) / ndotu
		Psi = w + si * ray_direction + plane_point

		circles_3d = np.append(circles_3d, Psi, axis = 0)
	
	return circles_3d

# --------- detect circles -----------
# 3D Points
circles3D, circles3D_reprojected = None, None
circles_grid_size = (6, 7)

cap = cv2.VideoCapture('ChAruco_Circles.webm')
validCaptures = 0

# Loop through images glob'ed
while cap.isOpened():
	ret, img = cap.read()
	if ret is False:
		break
	
	cv2.imshow('CircleBoard', img)
	if cv2.waitKey(1) == ord('q'):
		break

	# Grayscale the image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	""" Detect Circle Grid """
	params = cv2.SimpleBlobDetector_Params()
	params.filterByColor=1 
	params.blobColor=255
	params.filterByArea = True
	params.minArea=25
	params.maxArea=800
	# params.minDistBetweenBlobs=5

	# Set up the detector with default parameters.
	detector = cv2.SimpleBlobDetector_create(params)

	ret, circles = cv2.findCirclesGrid(gray, circles_grid_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector = detector)
	if ret is False or circles is None:
		continue

	# Draw Circles Centers
	img = cv2.drawChessboardCorners(img, circles_grid_size, circles, ret)

	""" Detect Aruco markers """
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
	# Refine detected markers
	# Eliminates markers not part of our board, adds missing markers to the board
	corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
			image = gray,
			board = CHARUCO_BOARD,
			detectedCorners = corners,
			detectedIds = ids,
			rejectedCorners = rejectedImgPoints,
			cameraMatrix = cameraMatrix,
			distCoeffs = distCoeffs)   
	img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 0, 255))
	
	# Only try to find CharucoBoard if we found markers
	if ids is not None and len(ids) >= 24:
		# Get charuco corners and ids from detected aruco markers
		response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
				markerCorners=corners,
				markerIds=ids,
				image=gray,
				board=CHARUCO_BOARD)
		# Estimate the posture of the charuco board, which is a construction of 3D space based on the 2D video 
		pose, rvec, tvec = aruco.estimatePoseCharucoBoard(
				charucoCorners=charuco_corners, 
				charucoIds=charuco_ids, 
				board=CHARUCO_BOARD, 
				cameraMatrix=cameraMatrix, 
				distCoeffs=distCoeffs)
		if pose:
			""" 
			Only use frames that reach this point! 
			If they reach here, that means we have a valid Circle grid and ChAruco pose
			"""
			imgpoints.append(circles)
			# Draw the camera posture calculated from the gridboard
			img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.1)

			""" Perform Ray-Plane Intersection to find 3D points """
			# ray-plane intersection: circle-center to chessboard-plane
			circles3D = intersectCirclesRaysToBoard(circles, rvec, tvec, cameraMatrix, distCoeffs)
			# objpoints.append(circles3D.astype('float32'))
			objpoints.append(objp)

			# re-project on camera for verification
			circles3D_reprojected, _ = cv2.projectPoints(circles3D, (0,0,0), (0,0,0), cameraMatrix, distCoeffs)
			imgpointsProj.append(circles3D_reprojected.astype('float32'))

			for c in circles3D_reprojected:
				cv2.circle(img, tuple(c.astype(np.int32)[0]), 3, (255,255,0), cv2.FILLED)

			validCaptures += 1
			if validCaptures == 40:
				break

			cv2.imshow('CircleBoard', img)
			if cv2.waitKey(0) == ord('q'):
				break

# Destroy any open CV windows
cap.release()
cv2.destroyAllWindows()

# calibrate projector
print("calibrate projector")
K_proj = np.zeros(shape=(3,3))
K_proj[0][0] = 1
K_proj[1][1] = 1
K_proj[0][2] = 1
K_proj[1][2] = 1
K_proj[2][2] = 1

ret, K_proj, dist_coef_proj, rvecs, tvecs = cv2.calibrateCamera(
	objpoints,
	imgpointsProj,
	gray.shape[::-1],
	None,
	None,
	flags=cv2.CALIB_FIX_INTRINSIC
)
print("proj calib mat after\n%s"%K_proj)
print("proj dist_coef %s"%dist_coef_proj.T)
print("calibration reproj err %s"%ret)

print("stereo calibration")
ret, K, dist_coef, K_proj, dist_coef_proj, proj_R, proj_T, _, _ = cv2.stereoCalibrate(
	objpoints,
	imgpoints,
	imgpointsProj,
	cameraMatrix,
	distCoeffs,
	K_proj,
	dist_coef_proj,
	gray.shape[::-1],
	flags=cv2.CALIB_FIX_INTRINSIC
)

proj_rvec, _ = cv2.Rodrigues(proj_R)

print("R \n%s"%proj_R)
print("T %s"%proj_T.T)
print("proj calib mat after\n%s"%K_proj)
print("proj dist_coef %s"	   %dist_coef_proj.T)
print("cam calib mat after\n%s" %K)
print("cam dist_coef %s"		%dist_coef.T)
print("reproj err %f"%ret)

f = open('./calibration/ProCamCalibration.pckl', 'wb')
pickle.dump((proj_R, proj_T.T, K_proj, dist_coef_proj.T, K, dist_coef.T, ret), f)
f.close()