#COS 470 by Xin Zhang
import cv2

# Open the video file
video = cv2.VideoCapture('video.mp4')

# Check if the video was successfully opened
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop until the end of the video or until a key is pressed
while True:
    # Capture frame-by-frame
    ret, frame = video.read()

    # If the frame was not retrieved successfully, break the loop
    if not ret:
        print("End of video.")
        break

    # Display the current frame
    cv2.imshow('Video Frame', frame)

    # Wait for 25ms and check if the user pressed 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("Exiting video playback.")
        break

# Release the VideoCapture object
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
