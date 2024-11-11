import cv2
import numpy as np

# ORB
img = cv2.imread('MountRushmore1.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (1200,600))

# Initiate ORB detector
orb = cv2.ORB_create()
 
# find the keypoints with ORB
kp = orb.detect(img,None)
 
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
 
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

# Display the image with keypoints
cv2.imshow('ORB Keypoints with Orientation', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()