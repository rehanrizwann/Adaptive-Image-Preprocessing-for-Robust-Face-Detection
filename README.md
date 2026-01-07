import cv2
import numpy as np

# Load input image
image = cv2.imread("image.jpg")
original_image = image.copy()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Analyze image contrast
contrast_mean = np.mean(gray_image)
contrast_std = np.std(gray_image)

# Apply histogram equalization if contrast is low
if contrast_std < 40:
    gray_image = cv2.equalizeHist(gray_image)

# Estimate noise level
noise_variance = np.var(gray_image)

# Apply adaptive noise reduction
if noise_variance > 500:
    gray_image = cv2.medianBlur(gray_image, 5)
else:
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Refine facial regions using morphological closing
morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, morph_kernel)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detect faces on raw image (baseline)
baseline_faces = face_cascade.detectMultiScale(
    cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY),
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Detect faces on preprocessed image
preprocessed_faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.05,
    minNeighbors=6,
    minSize=(30, 30)
)

# Draw rectangles for baseline faces (red)
for (x, y, w, h) in baseline_faces:
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Draw rectangles for preprocessed faces (green)
for (x, y, w, h) in preprocessed_faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Combine images side by side for comparison
comparison_image = np.hstack((original_image, image))

# Display results
cv2.imshow("Baseline (Red) vs Adaptive Preprocessing (Green)", comparison_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
