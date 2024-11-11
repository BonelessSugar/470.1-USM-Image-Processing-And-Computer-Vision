import cv2
import numpy as np

# BFMatcher ORB
img1 = cv2.imread('NotreDame.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
img1 = cv2.resize(img1, (600,600))
img2 = cv2.imread('NotreDame1.jpg',cv2.IMREAD_GRAYSCALE)         # trainImage
img2 = cv2.resize(img2, (600,600))
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('BFMatcher ORB', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()