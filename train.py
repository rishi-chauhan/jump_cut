import cv2
import os
import dipFuntions
from PIL import Image
# import re
import object_detection
from sklearn import linear_model

# convert text into int
# def atoi(text):
#     return int(text) if text.isdigit() else text
#
# # arrange in increasing order
# def natural_keys(text):
#     return [atoi(c) for c in re.split('(\d+)', text)]

# dictionary with name of video as key and jump cut timings in seconds as values in form of list of tuples. Each tuple is interval in which jump cut has taken place
cut_dict={"v1":[132,163,206,264,638,676,722,837,881,943,1027,1089,1127,1183,1259,1336,1422,1467,1504,1535,1554,1599,1652],
"v2":[27,86,142,196,241,276,327,356],
"v3":[135,482,870,1238,1631,2045],
"v4":[85,101,128,165,189,201,224,237,269],
"v5":[93,141,217,278,347,423,495,586,655,757,822,874,905,947,989,1046,1110,1145,1194,1224,1269,1306,1348,1378,1419,],
"v6":[263]}

x=[]
y=[]


# function that will train the regression model
def trainModel():
    # accessing each key in cut_dict
    for video in cut_dict:
        # get location of folder to be accessed
        frame_loc=video+"_frame/"

        # list of all frames
        # frame_list=os.listdir(frame_save_path)
        # frame_list.sort(key=natural_keys)
        # list to store all frames that have jump cut in between

        for cut in cut_dict[video]:
            req_frame=["frame"+str(cut)+".jpg","frame"+str(cut+1)+".jpg"]
            # print req_frame
            # frame1="frame"+str(cut)+".jpg"
            # frame2="frame"+str(cut+1)+".jpg"
            # req_frame.append(frame1)

            # obj_lst=[]
            obj_loc=[]
            # a list of histogram of all frames
            all_histogram=[]

            # process each frame in the folder and get its
            for frame in req_frame:
                # get the objects and their location in frame
                o,l=object_detection.getObject(frame_loc+frame)
                for i in xrange(1):
                    try:
                        if o[i]=="person":
                            obj_loc.append(l[i])
                    except:
                        pass

                # print "creating histogram"
                # current_file = os.path.join(frame_save_path, files)
                # open each frame in gray scale
                curent_frame=Image.open(os.path.join(frame_loc, frame)).convert("L")
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

            # x.append(hist_diff)
            if len(obj_loc)==2:
                x.append([hist_diff[0],abs(obj_loc[0][0]-obj_loc[1][0]),abs(obj_loc[0][1]-obj_loc[1][1])])
            else:
                x.append([hist_diff[0],240,240])
    y=[(1) for i in xrange(len(x))]
    classifier=linear_model.LinearRegression()
    return classifier.fit(x,y)

# print trainModel()
