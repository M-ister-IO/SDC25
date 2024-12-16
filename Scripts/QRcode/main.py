import cv2
from pyzbar.pyzbar import decode
import numpy as np
import yaml
import logging
import os
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to load YAML config
def load_config(config_path="config.yaml"):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

# Load config
config = load_config()

# Extract config
try:
    focal_length = config['focal_length']
    qr_code_side_length = config['qr_code_side_length']
except KeyError as e:
    logging.error(f"Missing configuration key: {e}")
    raise

def sort_corners(points):
    """
    Sort the corners of the QR code in a consistent order: top-left, top-right, bottom-right, bottom-left.
    """
    points = sorted(points, key=lambda p: p[1])  # Sort by y-coordinate
    top_two = sorted(points[:2], key=lambda p: p[0])  # Top-left and top-right
    bottom_two = sorted(points[2:], key=lambda p: p[0])  # Bottom-left and bottom-right
    return [top_two[0], top_two[1], bottom_two[1], bottom_two[0]]


def calculate_yaw(corners, qr_code_side_length=0.07, distance=0.3):
    """
    Calculate the yaw (rotation around the z-axis).
    Args:
        corners: List of four points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] sorted as top-left, top-right, bottom-right, bottom-left.
        qr_code_side_length: Actual side length of the QR code in meters.
        distance: Distance from the camera to the QR code in meters.
    Returns:
        yaw_angle: Yaw angle in degrees.
    """
    # Extract heights of the top and bottom sides
    top_left, top_right, bottom_right, bottom_left = corners

    left_height = np.linalg.norm(np.array(top_left) - np.array(bottom_left))
    right_height = np.linalg.norm(np.array(bottom_right) - np.array(top_right))

    tan_theta = (left_height - right_height) / (4*math.sqrt(left_height**2 + right_height**2 - (left_height - right_height)**2))
    
    # Calculate the yaw angle in radians
    theta_radians = math.atan(tan_theta)
    
    # Convert the yaw angle to degrees
    theta_degrees = math.degrees(theta_radians)

    return np.degrees(theta_degrees)

def calculate_center(points):
    """
    Calculate the center of the QR code based on its bounding box points.
    """
    center = np.mean(points, axis=0).astype(int)
    return tuple(center)

def draw_annotations(frame, center, tilt, points):
    """
    Draw bounding box, center, and annotations on the frame.
    """
    # Draw bounding box
    if abs(tilt)<=10:
        points_np = np.array(points, dtype=int)
        cv2.polylines(frame, [points_np], isClosed=True, color=(0, 255, 0), thickness=2)
    else:
        points_np = np.array(points, dtype=int)
        cv2.polylines(frame, [points_np], isClosed=True, color=(0, 0, 255), thickness=2)

    # Draw center point
    cv2.circle(frame, center, radius=5, color=(0, 0, 255), thickness=-1)

    # Add annotations
    cv2.putText(frame, f"Center: {(str(center[0]),str(center[1]))}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Yaw angle: {tilt:.0f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def process_video(source=0):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logging.error("Error: Could not open video source.")
        return

    logging.info("Starting video capture...")
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("No frame captured. Exiting...")
            break

        qr_codes = decode(frame)
        for qr_code in qr_codes:
            points = qr_code.polygon

            if len(points) == 4:  # Ensure a valid bounding box
                points_array = [(point.x, point.y) for point in points]
                sorted_points = sort_corners(points_array)

                center = calculate_center(sorted_points)
                z_rotation = calculate_yaw(sorted_points)

                draw_annotations(frame, center, z_rotation, sorted_points)

        # Show the frame
        cv2.imshow('QR Code rotation and Center', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Quitting video stream...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(0)
