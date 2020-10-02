import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


imgs = os.listdir("/home/hezi/final_project/final_project-master/datasets/test_lab7")
imgs.sort(key=natural_keys)
for i, img in enumerate(imgs):
    if img.endswith(".jpg"):
        im = cv2.imread("/home/hezi/final_project/final_project-master/datasets/test_lab7/"+img)
        im = im[0:4500, 0:4500]
        cv2.imwrite("/home/hezi/final_project/final_project-master/datasets/test_lab6/"+str(i+1)+".jpg",im)



plt.plot([1,2,3,4,5], [28.17,74.88,115.88,155.155,190.421], label = "sift + flann") #sift + flann
plt.plot([1,2,3,4,5], [26,48,75,100,130], label = "sift + flann + mask") # sift + flann + mask 
plt.plot([1,2,3,4,5], [9,20,29,46,61], label = "surf + flann + mask")    # surf + flann + mask 
plt.plot([1,2,3,4,5], [2.53,4.951,7.489,10.417,14.049], label = "sift + bfm + mask") # surf + bfm + mask
plt.ylabel('Time')
plt.xlabel('Image number')
plt.legend()
plt.show()


