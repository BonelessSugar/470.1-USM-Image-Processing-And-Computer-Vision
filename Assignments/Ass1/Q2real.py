#COS 470 by Taylor Brookes
import cv2
import numpy as np
import copy

# open the video file
video = cv2.VideoCapture('video.mp4')

# check if the videos were successfully opened
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# create master list of frames for video variants
frame_list = []
while True:
    check , vid = video.read()
    if not check:
        print("End")
        break
    # resize resolution so the final 2x2 grid can easily fit on my screen
    vid = cv2.resize(vid, (300,300))
    frame_list.append(vid)

# initialize an array of all the frames to be combined
frame_list_combined = []

# frame 1: YOUR_NAME watermark
# copy list instead of pointing to the same list
frame_list_1 = copy.deepcopy(frame_list)
for frame1 in frame_list_1:
    #modify frame 1
    img = frame1
    text = 'TAYLOR BROOKES'
    bottom_corner_pos = (100,275)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    colour = (0,0,0)
    thickness = 2
    frame1 = cv2.putText(img,text,bottom_corner_pos,font,scale,colour,thickness)
    # initialize, final will be [frame1, frame2, frame3, frame4]
    frame_list_combined.append([frame1,frame1,frame1,frame1])

# frame 2: reverse playback
frame_list_2 = copy.deepcopy(frame_list)
frame_list_2.reverse()
count = 0
for frame2 in frame_list_2:
    frame_list_combined[count][1] = frame2
    count += 1

# frame 3: grayscale
frame_list_3 = copy.deepcopy(frame_list)
count = 0
for frame3 in frame_list_3:
    frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    # back to BGR because grayscale is 2d and need 3D for .hstack because all frames have to be same dimensionality
    frame3 = cv2.cvtColor(frame3, cv2.COLOR_GRAY2BGR)
    frame_list_combined[count][2] = frame3
    count += 1

# frame 4: subtitles
#25fps (from mp4 properties), 336 frames
#2sec (f50) - 7sec (f199) = "this is the first sentence!"
#9sec (f225) - 11sec (f299) = "this is the second!"
frame_list_4 = copy.deepcopy(frame_list)
count = 0
for frame4 in frame_list_4:
    if 50 <= count < 200:
        img = frame4
        text = "this is the first sentence!"
        bottom_corner_pos = (50,275)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        colour = (0,0,0)
        thickness = 2
        frame4 = cv2.putText(img,text,bottom_corner_pos,font,scale,colour,thickness)
    elif 225 <= count < 300:
        img = frame4
        text = "this is the second!"
        bottom_corner_pos = (100,275)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        colour = (0,0,0)
        thickness = 2
        frame4 = cv2.putText(img,text,bottom_corner_pos,font,scale,colour,thickness)
    frame_list_combined[count][3] = frame4
    count += 1
# play frame_list_combined
for frame_array in frame_list_combined:

    # stack videos horizontally
    top_row = np.hstack((frame_array[0], frame_array[1]))
    bottom_row = np.hstack((frame_array[2], frame_array[3]))

    # stack the rows vertically
    joined_frame = np.vstack((top_row, bottom_row))

    # display the current frame
    cv2.imshow('Video Frame', joined_frame)
    # wait for 40ms (25fps per video properties) and check if the user pressed 'q' to quit
    if cv2.waitKey(40) & 0xFF == ord('q'):
        print("Exiting video playback.")
        break

# release the VideoCapture object
video.release()

# close all OpenCV windows
cv2.destroyAllWindows()
