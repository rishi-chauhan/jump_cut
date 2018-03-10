import cv2
import os
import dipFuntions
from PIL import Image
import re
import object_detection
import train
import test

# convert text into int
def atoi(text):
    return int(text) if text.isdigit() else text

# arrange in increasing order
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

# define everything
# the video to be processed
vidcap = cv2.VideoCapture('video/v7.mp4')
# path of the directory where extracted frames will be saved
frame_save_path="/home/rishi/Documents/sandbox/jump_cut/v7_frame/"
# reading the video
success,frames = vidcap.read()
# a list of all histograms of frames
all_histogram=[]

# list of all frames
frame_list=os.listdir(frame_save_path)
frame_list.sort(key=natural_keys)

# process each frame in the folder and get its
for frame in frame_list[:-1]:
    # print "creating histogram"
    # current_file = os.path.join(frame_save_path, files)
    # open each frame in gray scale
    curent_frame=Image.open(os.path.join(frame_save_path, frame)).convert("L")
    # get the array of image
    frame_array=dipFuntions.getImageArray(curent_frame)
    # get histogram of image and append it to all_histogram list
    all_histogram.append(dipFuntions.getHist(frame_array))

# take difference of consecutive frames
hist_diff=[]

for hist in xrange(0,len(all_histogram),2):
    d=[]
    try:
        for i in xrange(256):
            d.append(all_histogram[hist][i]-all_histogram[hist+1][i])
        hist_diff.append(d)
    # if number of frames are odd skip the last one
    except:
        pass

# nominalising hist_diff
for hist in xrange(len(hist_diff)):
    n=0
    for i in xrange(256):
        n+=abs(hist_diff[hist][i])
    hist_diff[hist]=n/256

# get objects
obj_loc=[]
x=[]
# y=[]
# process each frame in the folder and get its objects
for frame in frame_list[:-1]:
    # get the objects and their location in frame
    o,l=object_detection.getObject("v7_frame/"+frame)
    for i in xrange(1):
        try:
            if o[i]=="person":
                obj_loc.append(l[i])
        except:
            pass

# x.append(hist_diff)
obj_diff=[]
for i in xrange(0,len(obj_loc),2):
    obj_diff.append([abs(obj_loc[i][0]-obj_loc[i+1][0]),abs(obj_loc[i][1]-obj_loc[i+1][1])])

for i in xrange(len(hist_diff)):
    try:
        x.append([hist_diff[i],obj_diff[i][0],obj_diff[i][1]])
    except:
        x.append([hist_diff[i],240,240])

clf=test.trainModel()
result=clf.predict(x)
final=[]
for i in result:
    if i == 1:
        final.append(i)
print "Total number of jump cut: ", len(final)
print "\nJump cut between:"
for i in xrange(len(final)):
    print i+1,". ", i*2, " and ", (i*2)+1
