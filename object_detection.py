"""object detection module"""
import numpy
import cv2

PROTOTXT = "MobileNetSSD_deploy.prototxt.txt"
MODEL = "MobileNetSSD_deploy.caffemodel"
# frame="v4_frame/frame61.jpg"
PROBABILITY_THRESHOLD = 0.2

# initialize the list of class labels MobileNet SSD was trained to detect,
#then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# load model
NET = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

def get_object(frame):
    """returns object"""
	# list to store objects
    object_list = []
    object_loc = []
	# print "in getObject",frame
	# get the image and construct a blob for the image by resizing to a
	# fixed 300x300 pixels and then normalizing it
    image = cv2.imread(frame)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and predictions
    NET.setInput(blob)
    detections = NET.forward()

	# loop over the detections
    for i in numpy.arange(0, detections.shape[2]):
		# get the probability associated with the prediction
        probability = detections[0, 0, i, 2]

		# filtering out weak detections by ensuring the probability is greater than
		# the minimum threshold probability
        if probability > PROBABILITY_THRESHOLD:
			# get the index of the class label from the detections and then compute
			# the (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            obj = detections[0, 0, i, 3:7]*numpy.array([w, h, w, h])
            (x1, y1, x2, y2) = obj.astype("int")
            x = (x1+x2)/2
            y = (y1+y2)/2

			# display the prediction
            try:
                label = CLASSES[idx]
            except:
                pass
			# label = "{}: {:.2f}%".format(CLASSES[idx], probability * 100)
			# print("{}".format(label))

            object_list.append(label)
            object_loc.append((x,y))
        return object_list, object_loc

# print getObject(frame)
