# COS 470 by xin zhang
# this is the implementation of the harris corner algo

import cv2
import numpy as np

# Load image and convert to grayscale
image = cv2.imread('book.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Step 1: Compute gradients along the x and y directions
Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Step 2: Compute products of derivatives at every pixel
Ixx = Ix**2
Iyy = Iy**2
Ixy = Ix * Iy

# Step 3: Apply Gaussian filtering to the products of derivatives
Ixx = cv2.GaussianBlur(Ixx, (3,3), sigmaX=1)
Iyy = cv2.GaussianBlur(Iyy, (3,3), sigmaX=1)
Ixy = cv2.GaussianBlur(Ixy, (3,3), sigmaX=1)

k = 0.04
height, width = gray.shape

# Initialize the Harris response map
R = np.zeros_like(gray)

# Step 4: Calculate the Harris response R for each pixel
for y in range(1, height-1):
    for x in range(1, width-1):
        # Define the small window for M
        M = np.array([[Ixx[y, x], Ixy[y, x]], 
                      [Ixy[y, x], Iyy[y, x]]])

        # Calculate determinant and trace of the matrix M
        detM = np.linalg.det(M)
        traceM = np.trace(M)

        # Compute the Harris corner response
        R[y, x] = detM - k * (traceM**2)

# Step 5: Thresholding and Non-maximum Suppression
threshold = 0.01 * R.max()
corners = np.zeros_like(R, dtype=np.uint8)
corners[R > threshold] = 255

# Apply non-maximum suppression using dilation
dilated = cv2.dilate(corners, np.ones((3, 3), dtype=np.uint8))
final_corners = (corners == dilated) & (corners > 0)
print(np.argwhere(final_corners))
image[final_corners == True] = [0,255,0]

# Display the result
cv2.imshow('Harris Corners Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
