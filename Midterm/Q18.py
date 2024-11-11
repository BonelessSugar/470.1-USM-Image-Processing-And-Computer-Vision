'''
compile image manipulations into a video

generate 4 transformed images and compile the original + 4 transformed into a video with 5 frames
set frame rate of video to 1 frame per second

1. shifted image: translate image 50 pixels to right and 30 pixels down
2. rotated image: rotate 45 degrees around center without scaling
3. color-changed image: convert from BGR to Grayscale
4. text-overlayed image: place text "Transformed Image" on the image
    ensure text is at center-bottom of image and visible against background
'''
import cv2
import numpy as np
import copy
image = cv2.imread('dog.jpg')
image = cv2.resize(image, (900,600))

#1 shift
img1 = copy.deepcopy(image)
height, width = img1.shape[:2]
tx, ty = 50, 30
M = np.float32([[1,0,tx],[0,1,ty]])
transImg = cv2.warpAffine(img1, M, (width, height))
cv2.imshow('1. Move Right and Down', transImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

#2 rotate
img2 = copy.deepcopy(image)
height, width = img2.shape[:2]
center = (width // 2, height // 2)
angle = 45
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
rotatedImg = cv2.warpAffine(img2, M, (width, height))
cv2.imshow('2. Rotated 45deg', rotatedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

#3 color change
img3 = copy.deepcopy(image)
grayImg = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
#need this conversion back to BGR bc video is expecting 3 channels and doesnt show the 1 channel img
grayImgFORVIDEO = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR)
cv2.imshow('3. Grayscale', grayImgFORVIDEO)
cv2.waitKey(0)
cv2.destroyAllWindows()

#4 subtitle
img4 = copy.deepcopy(image)
text = "Transformed Image"
bottom_corner_pos = (150,550)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 2
colour = (255,255,255)
thickness = 2
textImg = cv2.putText(img4,text,bottom_corner_pos,font,scale,colour,thickness)
cv2.imshow('4. With Text', textImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

#5 make video
# define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# frames per second (1x speed, 0.5x speed executed with 2x frames later on)
frame_rate = 1.0
# frame size as width x height
frame_size = (900, 600)
# create the VideoWriter object
vidOut = cv2.VideoWriter('videoMid.avi', fourcc, frame_rate, frame_size)
#write all the frames
vidOut.write(image)
vidOut.write(transImg)
vidOut.write(rotatedImg)
vidOut.write(grayImgFORVIDEO)
vidOut.write(textImg)
vidOut.release()