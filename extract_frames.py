import cv2
import os

# the video to be processed
vidcap = cv2.VideoCapture('video/v7.mp4')

# reading the video
success,frames = vidcap.read()

# to name each frame starting from 1
count = 1
success = True

# saving frames in a folder
while success:
    success,frames = vidcap.read()
    print "Read a new frame: ", success
    # save frame as JPEG file
    cv2.imwrite(os.path.join("/home/rishi/Documents/sandbox/jump_cut/v7_frame", "frame%d.jpg") % count, frames)
    count += 1
