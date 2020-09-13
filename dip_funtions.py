"""Utiil functions for the project"""
import numpy
from PIL import Image

def get_image_array(image_object):
    """returns image array"""
    # get list of pixels
    pixel_values = list(image_object.getdata())
    # get image array
    image_arr = numpy.array(pixel_values)
    # reshape the array in the dimensions of image
    image_arr = image_arr.reshape(image_object.size[0], image_object.size[1])
    return image_arr

def get_hist(image_arr):
    """returns histogram of image"""
    row = image_arr.shape[0]    # get row of the image row
    col = image_arr.shape[1]    # get column of the image row
    hist = [0 for i in range(256)]
    for r in range(row):
        for c in range(col):
            hist[image_arr[r,c]] += 1
    return hist

def get_image_from_array(arr, width, height):
    """returns image from array"""
    arr = numpy.array(arr)
    # reshaping array
    arr = arr.reshape(width, height)
    arr = Image.fromarray(numpy.uint8(arr))
    return arr
