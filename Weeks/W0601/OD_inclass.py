import cv2
import numpy as np

model_path = "resnet50.onnx"
net = cv2.dnn.readNetFromONNX(model_path)

image_path = "puppy.jpg"
image = cv2.imread(image_path)

# resize image to 224*224 preprocessing
# first param is image
# 2nd is scale factor
# used for normalization, midigate noise from pixels
# want to go from 255 pixel value to 1 pixel value, map them from 0-255 to 0-1
# 3 resnet requires image to be 224*224
# 4 swapRB means swap RGB channel. OpenCV used BGR, resnet uses RGB
# small number is after normalization, large number (>100) is before normalization

blob = cv2.dnn.blobFromImage(image, scalefactor = 1/255.0, size = (224,224), mean = (0.485,0.456,0.406), swapRB=True)

net.setInput(blob)

# want to calculate the prediction
output = net.forward()

# output is possibility for each category
# for resnet, more than 2000 categories, so output will be array with more than 2000 elements
# each one is the corresponding to each class
# 0 to 100

print(output)


# output has 2000 categories, so find the one with the highest categories, detection results
predicted_class_id = np.argmax(output)
categories = []
# map class id to class name
with open("imagenet-classes.txt", 'r') as file:
    categories = [line.strip() for line in file.readlines()]

# use as a dictionary
# -1 is bc it starts w 1 for some reason, most images start with 0

predict_class_name = categories[predicted_class_id-1]
print(predict_class_name)
