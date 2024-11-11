# Work on text.jpg to recognize the text in the image.
# Write code to perform the recognition.
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('sign4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Morph open to remove noise and invert image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening


# If we have (x1,y1) as the top-left and (x2,y2) as the bottom-right
# ROI = image[y1:y2, x1:x2]
# sign1: 417,200 and 560,290
# sign2: 455,555 and 732,646
# sign3: 63,115 and 229,268
# sign4: 402,203 and 895,632

#roi = invert[203:632, 402:895]
roi = invert

# Perform text extraction
data = pytesseract.image_to_string(roi, lang='eng', config='--psm 6')
print(data)

#cv2.imshow('thresh', thresh)
#cv2.imshow('opening', opening)
#cv2.imshow('invert', invert)
cv2.imshow('roi', roi)
cv2.waitKey()