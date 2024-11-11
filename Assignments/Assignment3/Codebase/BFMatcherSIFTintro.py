import cv2
import numpy as np

# BFMatcher SIFT
img1 = cv2.imread('NotreDame.jpg',cv2.IMREAD_GRAYSCALE)  # queryImage
img1 = cv2.resize(img1, (600,600))
img2 = cv2.imread('NotreDame1.jpg',cv2.IMREAD_GRAYSCALE) # trainImage
img2 = cv2.resize(img2, (600,600))
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('BFMatcher SIFT', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()