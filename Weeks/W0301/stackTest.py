#COS 470 by Xin Zhang
import cv2
import numpy as np

# Load four images
image1 = cv2.imread('dog.jpg')

image1 = cv2.resize(image1, (300, 300))
image2 = image1
image3 = image1
image4 = image1

# Stack images horizontally
top_row = np.hstack((image1, image2))
bottom_row = np.hstack((image3, image4))

# Stack the rows vertically
joined_image = np.vstack((top_row, bottom_row))

# Display the joined image
cv2.imshow('Joined Image Grid', joined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the joined image
cv2.imwrite('joined_image_grid.jpg', joined_image)







