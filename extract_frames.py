"""module to extract frames"""
import os
import cv2

# the video to be processed
VIDCAP = cv2.VideoCapture('video/v7.mp4')

# reading the video
SUCCESS, FRAMES = VIDCAP.read()

# to name each frame starting from 1
COUNT = 1
SUCCESS = True

# saving frames in a folder
while SUCCESS:
    SUCCESS, FRAMES = VIDCAP.read()
    print("Read a new frame: ", SUCCESS)
    # save frame as JPEG file
    cv2.imwrite(os.path.join("/home/rishi/Documents/sandbox/jump_cut/v7_frame", "frame%d.jpg")
                % COUNT, FRAMES)
    COUNT += 1
