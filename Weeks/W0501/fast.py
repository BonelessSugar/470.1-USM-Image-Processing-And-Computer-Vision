# COS 470 by xin zhang
import cv2
import numpy as np

# Load image and convert to grayscale
image = cv2.imread('book.jpg')
image = cv2.resize(image, (1000,600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize FAST detector
fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)

# Detect corners
corners = fast.detect(gray, None)
cornerCoords = []
for corner in corners:
    cornerCoords.append([int(corner.pt[0]),int(corner.pt[1])])
print(cornerCoords)
# Draw corners on the image
output_image = cv2.drawKeypoints(image, corners, None, color=(255, 0, 0))

# Display the results
cv2.imshow('FAST Corners', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
