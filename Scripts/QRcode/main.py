import cv2
import numpy as np

def process_video(source=0):
    # Open video source (0 for camera or path for video file)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    qr_detector = cv2.QRCodeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from source.")
            break

        # Detect QR codes
        success, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)

        if success and points is not None:
            for i, polygon in enumerate(points):
                # Convert points to integers
                polygon = polygon.astype(int)

                # Draw the blue square around the QR code
                cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

                # Calculate the center of the QR code
                center_x = int(polygon[:, 0].mean())
                center_y = int(polygon[:, 1].mean())

                # Display center position in top-left corner
                cv2.putText(frame, f"Center: ({center_x}, {center_y})", (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Create a mask for the blue filter
                mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], (255, 0, 0))

                # Apply the blue filter
                blue_filter = np.zeros_like(frame, dtype=np.uint8)
                blue_filter[:, :, 0] = 255  # Blue channel
                frame = cv2.addWeighted(frame, 1.0, cv2.bitwise_and(blue_filter, mask), 0.5, 0)

        # Show the frame
        cv2.imshow('QR Code Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    source = "camera"
    if source.lower() == 'camera':
        process_video(0)  # Use webcam
    else:
        process_video(source)  # Use video file
