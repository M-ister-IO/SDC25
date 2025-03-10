import cv2
import numpy as np
import glob
import os

# Define the dimensions of the checkerboard
CHECKERBOARD = (6, 9)  # Number of inner corners per a chessboard row and column
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create vectors to store 3D points for each checkerboard image
objpoints = []
# Create vectors to store 2D points for each checkerboard image
imgpoints = []

# Define the real world coordinates for points in the checkerboard
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Get the file paths of the calibration images
images = glob.glob('calibration_images/*.jpg')

# Directory to save images with detected checkerboards
save_dir = 'calibrated_images'


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and save the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    image_path = os.path.join(save_dir, os.path.basename(fname))
    cv2.imwrite(image_path, img)
    print(f"Saved {image_path}")

# Perform camera calibration to get the camera matrix and distortion coefficients
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(dist_coeffs)