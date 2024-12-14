import cv2

# Known dimensions of the QR code (in meters)
S_actual = 0.1  # e.g., QR code side length is 10 cm
distance_to_qr = 1.0  # Distance from camera to QR code (in meters)

# Load the image containing the QR code
image = cv2.imread("qr_code_image.jpg")

# Detect the QR code
qr_detector = cv2.QRCodeDetector()
retval, points, _ = qr_detector.detectAndDecodeMulti(image)

if points is not None and len(points) > 0:
    # Calculate the apparent size of the QR code (in pixels)
    points = points[0]  # Assuming one QR code in the image
    S_apparent = cv2.norm(points[0] - points[1])  # Distance between two corners

    # Calculate the focal length
    focal_length = (distance_to_qr * S_apparent) / S_actual
    print(f"Apparent size (pixels): {S_apparent}")
    print(f"Estimated focal length (pixels): {focal_length}")
else:
    print("QR code not detected.")
