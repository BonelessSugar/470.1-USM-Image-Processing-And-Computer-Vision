import cv2
import numpy as np

def conv(image, kernel, padding, stride):
    imageHeight, imageWidth = image.shape
    kernelHeight, kernelWidth = kernel.shape
    #pads values to original image
    #4 padding means 1 row to top, 1 row to bottom, 1 col left, 1 col right, constant each padding, make it 0
    paddedImage = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    #make zero matrix
    output = np.zeros((imageHeight, imageWidth,), dtype=np.float32)
    #calc val for each pixel and modify array
    for y in range(imageHeight):
        for x in range(imageWidth):
            #region of interest
            #need an explanation on how this function part actually works
            roi = paddedImage[y*stride:y*stride+kernelHeight, x*stride:x*stride+kernelWidth]
            #y,x because numpy 1st para is col, 2nd para is row
            output[y,x] = (roi * kernel).sum()
    return output

image = np.array([
    [10, 10, 10, 10, 10, 10],
    [10, 10, 10, 255, 255, 10],
    [10, 10, 10, 255, 255, 10],
    [10, 10, 255, 255, 255, 10],
    [10, 10, 10, 255, 255, 10],
    [10, 10, 10, 10, 10, 10]
], dtype=np.uint8)

kernel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

kernel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

padding = 1
stride = 1

Gx = conv(image,kernel_x, padding, stride)
Gy = conv(image,kernel_y, padding, stride)
sobel = np.sqrt(Gx*Gx + Gy*Gy)
print(sobel)