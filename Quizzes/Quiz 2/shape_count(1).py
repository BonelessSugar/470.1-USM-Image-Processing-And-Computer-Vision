# COS470 by Xin Zhang

import cv2
import numpy as np

# Load the image
image = cv2.imread('shapes.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blurred, 50, 200)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

triangle_number = 0

for contour in contours:
    # Approximate the contour to reduce the number of points
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    # Draw the contours
    cv2.drawContours(image, [approx], 0, (0, 0, 0), 5)
    # Determine the shape based on the number of vertices in the approximated contour
    if len(approx) == 3:
        triangle_number += 1

print("how many triangles? " + str(triangle_number))
# Display the final image
cv2.imshow('Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



