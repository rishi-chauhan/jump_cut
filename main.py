"""Main file that gives the results"""
import os
import re
import test
from PIL import Image
import cv2
import dip_funtions
import object_detection
import train

def atoi(text):
    """converts text into int"""
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """arranges in increasing order"""
    return [atoi(c) for c in re.split('(\d+)', text)]

# define everything
# the video to be processed
VIDCAP = cv2.VideoCapture('video/v7.mp4')
# path of the directory where extracted frames will be saved
FRAME_SAVE_PATH = "/home/rishi/Documents/sandbox/jump_cut/v7_frame/"
# reading the video
SUCCESS, FRAMES = VIDCAP.read()
# a list of all histograms of frames
ALL_HISTOGRAM = []

# list of all frames
FRAME_LIST = os.listdir(FRAME_SAVE_PATH)
FRAME_LIST.sort(key=natural_keys)

# process each frame in the folder and get its
for frame in FRAME_LIST[:-1]:
    # print "creating histogram"
    # current_file = os.path.join(frame_save_path, files)
    # open each frame in gray scale
    curent_frame = Image.open(os.path.join(FRAME_SAVE_PATH, frame)).convert("L")
    # get the array of image
    frame_array = dip_funtions.get_image_array(curent_frame)
    # get histogram of image and append it to all_histogram list
    ALL_HISTOGRAM.append(dip_funtions.get_hist(frame_array))

# take difference of consecutive frames
HIST_DIFF = []

for hist in range(0, len(ALL_HISTOGRAM), 2):
    d = []
    try:
        for i in range(256):
            d.append(ALL_HISTOGRAM[hist][i]-ALL_HISTOGRAM[hist+1][i])
        HIST_DIFF.append(d)
    # if number of frames are odd skip the last one
    except:
        pass

# nominalising hist_diff
for hist in range(len(HIST_DIFF)):
    n = 0
    for i in range(256):
        n += abs(HIST_DIFF[hist][i])
    HIST_DIFF[hist] = n/256

# get objects
OBJ_LOC = []
X = []
# y=[]
# process each frame in the folder and get its objects
for frame in FRAME_LIST[:-1]:
    # get the objects and their location in frame
    o, l = object_detection.get_object("v7_frame/"+frame)
    for i in range(1):
        try:
            if o[i] == "person":
                obj_loc.append(l[i])
        except:
            pass

# x.append(hist_diff)
OBJ_DIFF = []
for i in range(0, len(OBJ_LOC), 2):
    OBJ_DIFF.append([abs(OBJ_LOC[i][0]-OBJ_LOC[i+1][0]), abs(OBJ_LOC[i][1]-OBJ_LOC[i+1][1])])

for i in range(len(HIST_DIFF)):
    try:
        X.append([HIST_DIFF[i], OBJ_DIFF[i][0], OBJ_DIFF[i][1]])
    except:
        X.append([HIST_DIFF[i], 240, 240])

CLF = test.trainModel()
RESULT = CLF.predict(X)
FINAL = []
for i in RESULT:
    if i == 1:
        FINAL.append(i)
print("Total number of jump cut: ", len(FINAL))
print("\nJump cut between:")
for i in range(len(FINAL)):
    print(i+1, ". ", i*2, " and ", (i*2)+1)
