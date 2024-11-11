# COS470 by Xin Zhang
import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('puppy.jpg', cv2.IMREAD_GRAYSCALE)


# Sobel:
# Apply Sobel operator to find the gradient in the x direction
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

# Apply Sobel operator to find the gradient in the y direction
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the gradient magnitude
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Convert magnitude to uint8
sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))

# Display the original and Sobel filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Sobel Magnitude', sobel_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Canny:
# Apply Gaussian blur to smooth the image and reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.2)

# Apply Canny edge detection
edges = cv2.Canny(blurred_image, 100, 200)

# Display the original and Canny edge images
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


