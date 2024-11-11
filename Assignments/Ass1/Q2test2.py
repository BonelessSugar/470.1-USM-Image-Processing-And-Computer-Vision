import cv2
import numpy as np

# Open the video file
video = cv2.VideoCapture('video.mp4')

# Check if the videos were successfully opened
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

#create list of frames for reverse order
frame_list = []
while True:
    check , vid = video.read()
    if not check:
        print("End")
        break
    vid = cv2.resize(vid, (300,300))
    frame_list.append(vid)

#play frame_list_combined
for frame_array in frame_list:

    # Stack videos horizontally
    top_row = np.hstack((frame_array, frame_array))
    bottom_row = np.hstack((frame_array, frame_array))

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

print("Hello")