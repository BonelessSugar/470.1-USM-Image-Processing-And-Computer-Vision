import cv2
import numpy as np
img1 = cv2.imread('Boston1.jpeg',cv2.IMREAD_GRAYSCALE)  # queryImage
#img1 = cv2.resize(img1, (600,600))
img2 = cv2.imread('Boston.jpeg',cv2.IMREAD_GRAYSCALE) # trainImage
#img2 = cv2.resize(img2, (600,600))
# Initiate SIFT detector
sift = cv2.SIFT_create(sigma=5.0)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters

#IMPORTANT FOR SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
goodMatches = 0
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        matchesMask[i]=[1,0]
        goodMatches += 1
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
cv2.imshow('FLANN SIFT', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# total matches
print(f"Total Matches: {len(matches):,}")
# good matches
print(f"Good Matches: {goodMatches:,}")
# bad matches
print(f"Bad Matches: {len(matches) - goodMatches:,}")
# accuracy
print(f"Accuracy: {goodMatches / len(matches) * 100:.2f}%")