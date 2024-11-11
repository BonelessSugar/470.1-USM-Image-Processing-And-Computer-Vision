import cv2
import numpy as np

# SIFT
img1 = cv2.imread('Boston.jpeg')
img1 = cv2.resize(img1, (600,600))
grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('Boston1.jpeg')
img2 = cv2.resize(img2, (600,600))
grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# Create SIFT object
sift = cv2.SIFT_create()
# Detect SIFT keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(grayImg1, None)
kp2, des2 = sift.detectAndCompute(grayImg2, None)
# Draw keypoints on the image with orientation
kpImg1 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kpImg2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

'''
# Good vs Bad matches
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
'''

# Display the image with keypoints
cv2.imshow('SIFT1 Keypoints with Orientation', kpImg1)
cv2.imshow('SIFT2 Keypoints with Orientation', kpImg2)
cv2.waitKey(0)
cv2.destroyAllWindows()