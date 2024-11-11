# COS 470 by xin zhang
# this is the template to implement the harris corner algo

import cv2
import numpy as np

# Load image and convert to grayscale
image = cv2.imread('dog.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Step 1: Compute gradients along the x and y directions
# hints: cv2.Sobel()
Ix = ???
Iy = ???

# Step 2: Compute Ix**2, Iyy**2, and Ix*Iy
Ixx = Ix**2
Iyy = Iy**2
Ixy = Ix * Iy

# Step 3: Apply Gaussian filtering to the products of derivatives
# hints: cv2.GaussianBlur()
Ixx = ???
Iyy = ???
Ixy = ???


# Initialize the Harris response map
R = np.zeros_like(gray)

# Step 4: Calculate the Harris response R for each pixel
height, width = gray.shape
# we directly discard the boundary pixels to avoid padding
for y in range(1, height-1):
    for x in range(1, width-1):
        # 4.1 Define the small window for M
        M = np.array([[Ixx[y, x], Ixy[y, x]], 
                      [Ixy[y, x], Iyy[y, x]]])

        # 4.2 Calculate determinant and trace of the matrix M
        # hints: calcualte det --> np.linalg.det(M)
        # calculate trace --> np.trace(M)
        detM = ???
        traceM = ???

        # 4.3 Compute the Harris corner response for pixel R[y, x]
        R[y, x] = ???

# Step 5: Thresholding and Non-maximum Suppression
# 5.1 thresholding:
threshold = 0.01 * R.max()
corners = np.zeros_like(R, dtype=np.uint8)
corners[R > threshold] = 255
# 5.2 non-Maximum suppresion
# hints: use cv2.dilate() for NMS 
dilated = cv2.dilate(corners, np.ones((3, 3), dtype=np.uint8))
final_corners = ???

image[final_corners == True] = [0,255,0]

# Display the result
cv2.imshow('Harris Corners Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
