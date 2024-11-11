# COS 470 CV by Xin Zhang
# use OpenCV + pretrained CNN for gender detection

#download pretained resnet from https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx

#0 --> male, 1-->female

import cv2
import numpy as np

# Load the ONNX model
model_path = 'gender_googlenet.onnx'
net = cv2.dnn.readNetFromONNX(model_path)

# Load the image
image_path = 'male.jpg'  # Replace with your image file path
image = cv2.imread(image_path)

if image is None:
    print("Could not read the image.")
    exit()

blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(224, 224), mean=(104, 117, 123), swapRB=False)

# Set the input to the model
net.setInput(blob)

# Predict the gender
output = net.forward()
print(output)
predicted_gender_id = np.argmax(output)
gender_classes = ['Male', 'Female']  # Define classes accordingly

predicted_gender = gender_classes[predicted_gender_id]

print(f"Predicted Gender: {predicted_gender}")

# Display the image with the prediction
cv2.putText(image, predicted_gender, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Image', image)
cv2.waitKey(0)  # Wait until any key is pressed
cv2.destroyAllWindows()
