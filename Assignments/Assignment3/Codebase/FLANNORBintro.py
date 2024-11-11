import cv2
import numpy as np
img1 = cv2.imread('Boston.jpeg',cv2.IMREAD_GRAYSCALE)  # queryImage
img1 = cv2.resize(img1, (600,600))
img2 = cv2.imread('Boston1.jpeg',cv2.IMREAD_GRAYSCALE) # trainImage
img2 = cv2.resize(img2, (600,600))
# Initiate SIFT detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# FLANN parameters

# IMPORTANT FOR ORB
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
# Match descriptors.
matches = flann.match(des1,des2)

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,**draw_params)
cv2.imshow('FLANN ORB', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()