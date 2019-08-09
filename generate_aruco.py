import numpy as np
import cv2
import cv2.aruco as aruco

# Select type of aruco marker (size)
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)

# Create an image from the marker
# second param is ID number
# last param is total image size
img = aruco.drawMarker(aruco_dict, 1, 50)
cv2.imwrite("test_marker1.jpg", img)
img = aruco.drawMarker(aruco_dict, 2, 50)
cv2.imwrite("test_marker2.jpg", img)
img = aruco.drawMarker(aruco_dict, 3, 50)
cv2.imwrite("test_marker3.jpg", img)
img = aruco.drawMarker(aruco_dict, 4, 50)
cv2.imwrite("test_marker4.jpg", img)

# image = np.zeros((480, 640, 3), np.uint8)
# image[:] = (255, 255, 255)

# height = imgL.shape[0] 
# width = imgL.shape[1]

# imgL[ 0:img.shape[0], 0:img.shape[1]] = img
# imgL[ 0:img.shape[0], 0-img.shape[1]:width] = img
# imgL[ 0-img.shape[0]:height, 0:img.shape[1]] = img
# imgL[ 0-img.shape[0]:height, 0-img.shape[1]:width] = img


# # Display the image to us
# cv2.imshow('frame', imgL)

# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()