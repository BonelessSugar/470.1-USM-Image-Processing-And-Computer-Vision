#COS 470 by Xin Zhang
import cv2
import numpy as np

# Load an image
image = cv2.imread('puppy.jpg')
# Get the dimensions of the image
height, width = image.shape[:2]

# 1. translation
# Define the translation matrix
tx, ty = 50, 100  # Translation offsets
M = np.float32([[1, 0, tx], [0, 1, ty]])

# Apply the translation using cv2.warpAffine
# (width, height) specifies the size of the translated image
translated_image = cv2.warpAffine(image, M, (width, height))

# Display the original and translated images
cv2.imshow('Original Image', image)
cv2.imshow('Translated Image', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# 2. rotate
# Define the center of the image for rotation
center = (width // 2, height // 2)

# Define the rotation angle and scale
angle = 45  # Rotate 45 degrees clockwise
scale = 1.0  # No scaling

# Get the rotation matrix using cv2.getRotationMatrix2D
M = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the rotation using cv2.warpAffine
rotated_image = cv2.warpAffine(image, M, (width, height))

# Display the original and rotated images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. affine
# Define the source points (points in the original image)
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])

# Define the destination points (where the points should be mapped to in the output image)
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# Calculate the affine transformation matrix
M = cv2.getAffineTransform(pts1, pts2)

# Apply the affine transformation using cv2.warpAffine
affine_transformed_image = cv2.warpAffine(image, M, (width, height))

# Display the original and transformed images
cv2.imshow('Original Image', image)
cv2.imshow('Affine Transformed Image', affine_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 4. perspective
# Define the source points (points in the original image)
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

# Define the destination points (where the points should be mapped to in the output image)
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

# Calculate the perspective transformation matrix
M = cv2.getPerspectiveTransform(pts1, pts2)

# Get the dimensions of the output image (the width and height of the destination rectangle)
width, height = 300, 300

# Apply the perspective transformation using cv2.warpPerspective
perspective_transformed_image = cv2.warpPerspective(image, M, (width, height))

# Display the original and transformed images
cv2.imshow('Original Image', image)
cv2.imshow('Perspective Transformed Image', perspective_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()











