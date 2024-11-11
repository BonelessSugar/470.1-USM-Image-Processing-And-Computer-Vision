'''
white rectangle with known dimensions (h:28.6cm, w:26.1cm)
white rectangle is enclosed around a black background
red circle inside white rectangle
use OpenCV to estimate the area of the red circle in cm^2
'''
'''
first reduce image to white rectangle
find the number of pixels the white rectangle is in width
then scan each line width-wise for the line with the most red pixels
then take that red circle diameter, divide by 2 to get radius, convert to cm, find area
pi*r^2
'''
import cv2
import numpy as np

scale = 30

# Load the image
image = cv2.imread('objectMeasure.png')
cv2.imshow('Initial Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blurred, 75, 200)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # Approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # Perspective transform if the contour has 4 points (likely the paper and not book because using RETR_EXTERNAL)
    if len(approx) == 4:

        # Get the points for perspective transform
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # Order the points: top-left, top-right, bottom-right, bottom-left
        # pts is the array of 4 corners
        # bottom right corner is largest
        # top left corner is smallest
        # 0 is minimum
        # 2 will be largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # y - x
        # 1 will be top right, minimum
        # 3 will be bottom left, maximum
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Set desired size and aspect ratio for the cards
        width = round(26.1 * scale)
        height = round(28.6 * scale)

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(image, M, (width, height))

        # Show the result
        
        cv2.imshow('Just White Rectangle', warp)
        cv2.waitKey(0)
cv2.destroyAllWindows()

#NEXT FIND THE LINE WITH THE MAX AMOUNT OF RED PIXELS
height, width, channels = warp.shape
print("New white rectangle dimensions: " + str(height) + ", " + str(width) + " pixels.")
#print(warp[500,500])
#print(warp[y,x])
totalRed = 0
totalPixel = height * width
for x in range(0,height):
    for y in range(0, width):
        if warp[x,y][0] < 255:
            totalRed += 1
print(f"Total image area: {totalPixel:,} pixels.")
print(f"Red circle area: {totalRed:,} pixels.")
#totalArea = 26.1cm * 28.6cm = 746.46cm^2
#circleArea = totalArea / (totalPixel / totalRed)
#=746.46 / (671814/70295)
#=78.1
print(f"Red circle area in cm^2: {((28.6 * 26.1) / (totalPixel / totalRed)):,.2f}")
