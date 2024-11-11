#COS 470 by Taylor Brookes
import cv2

# open the video file
video = cv2.VideoCapture('video.mp4')

# check if the video was successfully opened
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# loop until the end of the video or until a key is pressed
count = 0
while True:
    # capture frame-by-frame
    ret, frame = video.read()

    # if the frame was not retrieved successfully, break the loop
    if not ret:
        print("End of video.")
        break

    # display only half the frames for 2x speed
    if count % 2 == 0:
        count += 1
        cv2.imshow('Video Frame', frame)
        continue
    count += 1

    # wait for 40ms (25fps per video properties) and check if the user pressed 'q' to quit
    if cv2.waitKey(40) & 0xFF == ord('q'):
        print("Exiting video playback.")
        break

# release the VideoCapture object
video.release()

# close all OpenCV windows
cv2.destroyAllWindows()
