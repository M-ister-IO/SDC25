import cv2
import numpy as np

# Load camera calibration parameters
CAMERA_MATRIX = np.array([
    [260.35815276, 0., 360.95956209],
    [0., 255.4228265, 224.57304735],
    [0., 0., 1.]
])

DIST_COEFFS = np.array([-0.19192441, 0.25063116, 0.00799991, 0.03010645, -0.07408033])

# Define the Aruco dictionary & marker size
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
MARKER_SIZE = 0.07  # Square size in meters (adjust to your real size)
PARAMETERS = cv2.aruco.DetectorParameters()

def detect_aruco(source=1, output_file='output.avi'):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to grayscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(grey, ARUCO_DICT, parameters=PARAMETERS)

        if ids is not None:

            # Estimate pose for each detected marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS
            )

            for i in range(len(ids)):
                # Draw marker outline and axes
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                # cv2.aruco.drawAxis(frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[i], tvecs[i], 0.03)

                # Extract rotation and translation
                rvec = rvecs[i]
                tvec = tvecs[i]

                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                # Get rotation angles (roll, pitch, yaw)
                roll = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi  # Rotation around X-axis
                pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)) * 180 / np.pi  # Y-axis
                yaw = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi  # Z-axis

                # Determine the color based on the pitch angle
                color = (0, 255, 0) if abs(pitch) < 2.5 else (0, 0, 255)  # Green if pitch is close to 0, otherwise red

                # Draw a rectangle around the marker
                int_corners = np.int32(corners[i].reshape(-1, 2))
                cv2.polylines(frame, [int_corners], isClosed=True, color=color, thickness=2)

                # Display rotation angles
                cv2.putText(frame, f"ID {ids[i]}: Roll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f}",
                            (10, 30 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Write the frame to the output file
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved as {output_file}")

# Run detection
if __name__ == "__main__":
    detect_aruco(2, 'output.avi')  # Use webcam (or replace 0 with a video file)
