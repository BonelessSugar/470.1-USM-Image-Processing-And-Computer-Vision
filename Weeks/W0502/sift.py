#Code example for SIFT, COS470/570 by Xin

import cv2
import numpy as np

# Load the image
image = cv2.imread('test.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create SIFT object
sift = cv2.SIFT_create()

# Detect SIFT keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

# can you print the descriptor, and discuss your observations?

# Draw keypoints on the image with orientation
keypoint_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
cv2.imshow('SIFT Keypoints with Orientation', keypoint_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
