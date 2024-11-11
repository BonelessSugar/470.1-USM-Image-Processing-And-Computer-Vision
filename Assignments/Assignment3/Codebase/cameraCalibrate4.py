import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
imgL = cv.imread('Castle.jpg', cv.IMREAD_GRAYSCALE)
imgL = cv.resize(imgL, (600,600))
imgR = cv.imread('Castle1.jpg', cv.IMREAD_GRAYSCALE)
imgR = cv.resize(imgR, (600,600))
 
stereo = cv.StereoBM.create(numDisparities=16, blockSize=5)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()