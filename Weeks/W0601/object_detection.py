# COS 470 CV by Xin Zhang
# use OpenCV + pretrained resnet for object detection

#download pretained resnet from https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx
#download classes.txt from https://github.com/BigWZhu/ResNet50/blob/master/imagenet-classes.txt?ref=blog.roboflow.com

import cv2
import numpy as np

# Load the ONNX model
model_path = 'resnet50.onnx'
net = cv2.dnn.readNetFromONNX(model_path)

# Load the image
image_path = 'puppy.jpg'
image = cv2.imread(image_path)

# Preprocess the image
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(224, 224), mean=(0.485, 0.456, 0.406), swapRB=True)

# Set the input to the model
net.setInput(blob)

# Predict the class
output = net.forward()
print(output)

# Output interpretation
predicted_class_id = np.argmax(output)
categories = []
with open("imagenet-classes.txt", 'r') as file:
    categories = [line.strip() for line in file.readlines()]

predicted_class = categories[predicted_class_id-1]

print(f"The image is a: {predicted_class}")

# Display the image with the prediction
cv2.putText(image, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
