#COS 470 by Xin Zhang
# how to use haar cascade to detect eye blink

import cv2

def detect_blink(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detect faces in the frame

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 12)  # Detect eyes in the face region

        # Check for blinks: 
        # if less than 2 eyes are detected, it might be a blink
        if len(eyes) < 2:
            cv2.putText(frame, "Blink Detected!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        

if __name__ == '__main__':
    # Load the pre-trained Haar Cascade models for face and eyes
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)  # Start video capture from the default camera

    while True:
        ret, frame = cap.read()  # Read a frame from the video capture
        if not ret:
            print("Error: Unable to read from video capture.")
            break

        detect_blink(frame, face_cascade, eye_cascade)  # Detect blinks in the frame

        cv2.imshow('Eye Blink Detection', frame)  # Display the frame with detection results

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows
