# Simeon Cromarty Master's Project - Camera Calibration
# Functions used from OpenCV Library documents (https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
import numpy as np
import cv2
import glob

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Set chessboard size
checkerboard_dims = (7, 9) 
objp = np.zeros((checkerboard_dims[0]*checkerboard_dims[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Load calibration images
images = glob.glob('VisionProject/calibration images/*.png')

# Read each image in calibration folder
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

    # If found, add object and image points
    if ret:
        objpoints.append(objp)

        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(refined_corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, checkerboard_dims, refined_corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('mtx = ',mtx)
print('dist = ', dist)
print('rvecs = ', rvecs)
print('tvecs = ', tvecs)

# Function to undistort images
def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_mtx)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    
    print(camera_matrix)
    print(dist_coeffs)
    return undistorted_img

# Show undistorted image 
new_img = cv2.imread('VisionProject/calibration images/opencv_frame_50.png')
for fname in new_img:
    undistorted_new_img = undistort_image(new_img, mtx, dist)
    cv2.imshow('Undistorted Image', undistorted_new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()