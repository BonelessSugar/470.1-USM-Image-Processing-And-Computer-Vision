#COS 470 by Xin Zhang
import cv2
import numpy as np

# Load a grayscale image
image = cv2.imread('puppy.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (100, 50))
# 1. simple threshold
# Apply simple thresholding
threshold_value = 127
retval, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Display the original and thresholded images
print("Threshold Value Used:", retval)
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. adaptive threshold
# Apply adaptive thresholding
adaptive_binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)

# Display the original and thresholded images
cv2.imshow('Original Image', image)
cv2.imshow('Adaptive Thresholding', adaptive_binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 3. OTSU
# Apply Otsu's thresholding
retval, otsu_binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the original and Otsu thresholded images
print("Otsu's Optimal Threshold Value:", retval)
cv2.imshow('Original Image', image)
cv2.imshow('Otsu Thresholding', otsu_binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()












