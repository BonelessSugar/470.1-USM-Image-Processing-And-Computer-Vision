'''
step 1: transform paper from trapezoid to rectangle and remove border from paper
        paper = 27.8cm x 21.5cm
        SCALE = 30 (done so it can fit on my computer screen)
step 2: locate corners of book on no-bordered image that is just paper and book
step 3: measure from corner to corner and average the results
step 4: calculate book dimensions in cm
        book = (pixel / SCALE)cm x (pixel / SCALE)cm
'''

import cv2
import numpy as np

# QUESTION 1.1: MEASURE DIMENSIONS OF BOOK

# STEP 1: TRANSFORM PAPER

scale = 30

# Load the image
image = cv2.imread('book.jpg')

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
        width = round(27.8 * scale)
        height = round(21.5 * scale)

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
        '''
        cv2.imshow('Warped Card', warp)
        cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# STEP 2: FIND CORNERS OF BOOK

# Load the image
image = warp

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blurred, 75, 200)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#array: widthTop, widthBottom, widthAvg, lengthLeft, lenthRight, lengthAvg
dimensionArray = []

# Loop over the contours
for contour in contours:
    # Approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # It's the book if it has 4 points
    if len(approx) == 4:

        # Get the corners
        pts = approx.reshape(4, 2)

        # pts orders the corners as (CCW): top, left, bottom, right
        # top(TR) = smallest y
        # left(TL) = smallest x
        # bottom(BL) = biggest y
        # right(BR) = biggest x

# STEP 3: MEASURE FROM CORNER TO CORNER

        # widthTop
        # top X to left X, TX > LX
        legX = pts[0][0] - pts[1][0]
        # top Y to left Y, LY > TY
        legY = pts[1][1] - pts[0][1]
        hyp = np.hypot(legX, legY)
        dimensionArray.append(hyp)

        #widthBottom
        legX = pts[2][0] - pts[3][0]
        legY = pts[3][1] - pts[2][1]
        hyp = np.hypot(legX, legY)
        dimensionArray.append(hyp)

        #widthAvg
        dimensionArray.append((dimensionArray[0] + dimensionArray[1]) / 2)

        #lengthLeft
        # bottom X - left X
        legX = pts[2][0] - pts[1][0]
        # bottom Y - left Y
        legY = pts[2][1] - pts[1][1]
        hyp = np.hypot(legX, legY)
        dimensionArray.append(hyp)

        #lengthRight
        # right X - top X
        legX = pts[3][0] - pts[0][0]
        # right Y - top Y
        legY = pts[3][1] - pts[0][1]
        hyp = np.hypot(legX, legY)
        dimensionArray.append(hyp)

        #lengthAvg
        dimensionArray.append((dimensionArray[3] + dimensionArray[4]) / 2)

# STEP 4: PIXEL TO CM
dimWidth = round(dimensionArray[2] / scale, 2)
dimHeight = round(dimensionArray[5] / scale, 2)
theAnnotation = "Book dimensions: " + str(dimWidth) + "cm by " + str(dimHeight) + "cm"
print(theAnnotation)

# QUESTION 1.2: DRAW RECTANGE AROUND BOOK, DISPLAY BOOK WIDTH AND HEIGHT ON THE IMAGE

# Load the image
image = cv2.imread('book.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blurred, 50, 200)

# Find contours
# not sure why this works but it does. CCOMP is the mode.
contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# same with below
for i in range(len(contours)):
    # same with below
    cv2.drawContours(image, contours, 0, (0, 255, 0), 5)
    # this is taken from Q2 video assignment 1 and slightly edited
    img = image
    text = theAnnotation
    annotate_pos = (300,175)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    colour = (0,0,255)
    thickness = 2
    frame1 = cv2.putText(img,text,annotate_pos,font,scale,colour,thickness)

# Display the final image
cv2.imshow('Book', image)
cv2.waitKey(0)
cv2.destroyAllWindows()