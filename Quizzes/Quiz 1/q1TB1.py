import cv2
import numpy as np

image = cv2.imread('dog.jpg')

#convert image from BGR to grayscale
#save as gray_image.jpg
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.jpg', gray_img)

#resize image to a smaller one (width = 0.5 * original, height = 0.5 * original)
#save as resized_image.jpg
height, width = image.shape[:2]
resized_img = cv2.resize(image, (width / 2, height / 2))
cv2.imwrite('resized_image.jpg', resized_img)

#rotate image by 90 degrees clockwise
#save as rotated_image.jpg
center = (width // 2, height // 2)
angle = 90
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated_img = cv2.warpAffine(image, M, (width, height))
cv2.imwrite('rotated_image.jpg', rotated_img)

#flip image horizontally
#save as flipped_image.jpg
flipped_img = cv2.flip(image, 1)
cv2.imwrite('flipped_image.jpg', flipped_img)

#draw green rectangle top left (1000,250) and bottom right (1400,600)
#add text "dog" immediately right of rectange's top-right corner, colour (0,255,0)
#save image as annotated_image.jpg
image = np.ones((2000, 2000, 3), dtype=np.uint8) * 255
cv2.rectangle(image, (1000, 250), (1400, 600), (0, 255, 0), -1)
cv2.putText(image, 'OpenCV-House', (1400, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
cv2.imwrite('annotated_image.jpg', image)