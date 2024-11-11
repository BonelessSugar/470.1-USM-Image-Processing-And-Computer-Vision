import cv2
import numpy as np

# Open the video file
video = cv2.VideoCapture('video.mp4')

# Check if the videos were successfully opened
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop until the end of the video or until a key is pressed
while True:
    # Capture frame-by-frame
    ret, frame1 = video.read()

    # If the frame was not retrieved successfully, break the loop
    if not ret:
        print("End of video.")
        break

    # Resize videos so they can all fit on my screen
    frame1 = cv2.resize(frame1, (300,300))
    frame2 = cv2.resize(frame1, (300,300))
    frame3 = cv2.resize(frame1, (300,300))
    frame4 = cv2.resize(frame1, (300,300))

    #modify frame 1
    img = frame1
    text = 'TAYLOR BROOKES'
    bottom_corner_pos = (100,275)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    colour = (0,0,0)
    thickness = 2
    frame1 = cv2.putText(img,text,bottom_corner_pos,font,scale,colour,thickness)

    #modify frame 2

    # Stack videos horizontally
    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((frame3, frame4))

    # Stack the rows vertically
    joined_frame = np.vstack((top_row, bottom_row))

    # Display the current frame
    cv2.imshow('Video Frame', joined_frame)

    # Wait for 25ms and check if the user pressed 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("Exiting video playback.")
        break

# Release the VideoCapture object
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
