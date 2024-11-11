#COS 470 by Taylor Brookes
import cv2

# open the video file
video = cv2.VideoCapture('video.mp4')
# define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
# frames per second (1x speed, 0.5x speed executed with 2x frames later on)
frame_rate = 25.0  
# frame size as width x height
frame_size = (int(video.get(3)), int(video.get(4)))  
# create the VideoWriter object
out = cv2.VideoWriter('video0303.avi', fourcc, frame_rate, frame_size)

# check if the video was successfully opened
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# loop until the end of the video or until a key is pressed
while True:
    # capture frame-by-frame
    ret, frame = video.read()
    # if the frame was captured successfully
    if not ret:
        break
    # write the frame to the video file twice for 1/2 speed
    out.write(frame)
    out.write(frame)

# release the VideoCapture and VideoWriter objects
video.release()
out.release()

# close all OpenCV windows
cv2.destroyAllWindows()
