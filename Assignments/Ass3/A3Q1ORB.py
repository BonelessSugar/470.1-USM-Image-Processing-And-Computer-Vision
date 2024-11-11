import cv2
import numpy as np

# ORB
img1 = cv2.imread('Boston.jpeg')
img1 = cv2.resize(img1, (600,600))
grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('Boston1.jpeg')
img2 = cv2.resize(img2, (600,600))
grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# Create ORB object
orb = cv2.ORB_create()
# Detect ORB keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(grayImg1, None)
kp2, des2 = orb.detectAndCompute(grayImg2, None)
# Draw keypoints on the image with orientation
kpImg1 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kpImg2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Display the image with keypoints
cv2.imshow('ORB1 Keypoints with Orientation', kpImg1)
cv2.imshow('ORB2 Keypoints with Orientation', kpImg2)
cv2.waitKey(0)
cv2.destroyAllWindows()