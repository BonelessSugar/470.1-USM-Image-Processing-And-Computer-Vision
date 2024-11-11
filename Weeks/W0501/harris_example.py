import cv2
import numpy as np

# Load image and convert to grayscale
image = cv2.imread('book.jpg')  # Adjust path as needed
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to float32
gray = np.float32(gray)

# Parameters for Harris corner detection
block_size = 2  # Size of the neighbourhood considered for corner detection
ksize = 3       # Aperture parameter of the Sobel derivative used
k = 0.06        # Harris detector free parameter in the equation

# Detecting corners
harris_response = cv2.cornerHarris(gray, block_size, ksize, k)

# Threshold for an optimal value; it may vary depending on the image
threshold = 0.01 * harris_response.max()

# Apply non-maximum suppression to refine the corners
# Create a mask to mark the locations of true corners
N = 3  # Neighborhood size for non-max suppression (3x3)
mask = np.zeros_like(harris_response)
mask[harris_response > threshold] = 255

# Perform dilation to isolate the maximum points
dilated_mask = cv2.dilate(mask, np.ones((N, N), dtype=np.uint8))

# Compare dilated mask with the original mask to find local maxima
final_corners = (mask == dilated_mask) & (mask > 0)
print(np.argwhere(final_corners))
image[final_corners == True] = [0,255,0]

# Display the result
cv2.imshow('Harris Corners Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
