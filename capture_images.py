import cv2
import os
import time

# Directory to save captured images
save_dir = 'calibration_images'


# Open the USB camera (usually device 0 or 1)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Capturing images. Press Ctrl+C to stop.")

image_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Save the captured image automatically every few seconds
        image_path = os.path.join(save_dir, f"calibration_image_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Captured {image_path}")
        image_count += 1

        time.sleep(3)  # Wait for 1 second between captures

except KeyboardInterrupt:
    print("Image capture stopped by user.")

cap.release()
print("Camera released.")