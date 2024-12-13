import cv2
from pyzbar.pyzbar import decode
import numpy as np

def sort_corners(points):
    """
    Sort the corners of the QR code in a consistent order: top-left, top-right, bottom-right, bottom-left.
    """
    points = sorted(points, key=lambda p: p[1])  # Sort by y-coordinate
    top_two = sorted(points[:2], key=lambda p: p[0])  # Top-left and top-right
    bottom_two = sorted(points[2:], key=lambda p: p[0])  # Bottom-left and bottom-right
    return [top_two[0], top_two[1], bottom_two[1], bottom_two[0]]

def calculate_z_orientation(points):
    """
    Estimate the z-orientation (tilt) of the QR code based on side heights.
    """
    # Ensure points are sorted in the correct order
    sorted_points = sort_corners(points)
    top_left, top_right, bottom_right, bottom_left = sorted_points

    # Calculate side heights
    height_left = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
    height_right = np.linalg.norm(np.array(bottom_right) - np.array(top_right))

    # Calculate height difference and average height
    height_diff = abs(height_left - height_right)
    avg_height = (height_left + height_right) / 2

    # Calculate tilt angle in degrees
    tilt_angle = np.arctan2(height_diff, avg_height) * 180 / np.pi
    return tilt_angle, height_left, height_right

def calculate_center(points):
    """
    Calculate the center of the QR code based on its bounding box points.
    """
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    center_x = sum(x_coords) // len(x_coords)
    center_y = sum(y_coords) // len(y_coords)
    return center_x, center_y

def process_video(source=0):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        qr_codes = decode(frame)
        for qr_code in qr_codes:
            points = qr_code.polygon

            if len(points) == 4:  # Ensure a valid bounding box
                # Convert points to a usable format
                points_array = [(point.x, point.y) for point in points]
                sorted_points = sort_corners(points_array)

                # Calculate center
                center_x, center_y = calculate_center(sorted_points)

                # Calculate z-orientation
                z_orientation, height_left, height_right = calculate_z_orientation(sorted_points)

                # Display z-orientation and height info
                cv2.putText(frame, f"Z-Tilt: {z_orientation:.2f}°", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"H-L: {height_left:.1f}, H-R: {height_right:.1f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw the bounding box and display information
                points_np = np.array(sorted_points, dtype=int)
                cv2.polylines(frame, [points_np], isClosed=True, color=(255, 0, 0), thickness=2)


                cv2.putText(frame, f"Center: ({center_x}, {center_y})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw the center point
                cv2.circle(frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

        # Show the frame
        cv2.imshow('QR Code Orientation and Center', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    source = "camera"
    if source.lower() == "camera":
        process_video(0)
    else:
        process_video(source)