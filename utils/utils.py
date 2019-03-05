"""
    CRUNCH all the utility functions in here
"""
from __future__ import division  
import os
import cv2
import copy
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import gaussian, median
"""
=============================================================
"""

def read_images(folder='images'):
    folder = folder + "/"
    print("Reading images from {0}".format(folder))
    list_images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith(".jpg")]
    list_img = []
    list_img_name = []
    for i in list_images:
        path = folder + i
        print(path)
        list_img_name.append(path.split('/')[1])
        img = plt.imread(path)
        rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
        list_img.append(rgb_img)
    return list_img, list_img_name, img.shape

def filter_image(filter, image):
    if filter== "MEDIAN":
        print("Using {0} filter....".format(filter.upper()))
        filtered_image = median(image, disk(10))
    elif filter.upper() == "GAUSSIAN":
        print("Using {0} filter....".format(filter.upper()))
        filtered_image = gaussian(image, sigma=1, multichannel=True)

    return filtered_image

def clustering(labels,image, cluster):
    image_boundaries = get_image_boundaries(labels, image.shape)

    clustering = np.reshape(np.array(labels, dtype=np.uint8),
            (image.shape[0], image.shape[1]))

    sorted_labels = sorted([n for n in range(cluster)],
        key=lambda x: -np.sum(clustering == x))
    
    segmented_image = np.zeros(image.shape[:2], dtype=np.uint8)

    for i, label in enumerate(sorted_labels):
        segmented_image[clustering == label] = int(255 / (cluster + 1)) * i

    return segmented_image, image_boundaries


def get_image_boundaries(labels, size):
    height = size[0]
    width = size[1]
    ret = np.zeros([height,width, 1], dtype=bool)
    div = labels.reshape([height,width,1])
    df0 = np.diff(div,axis=0)
    df1 = np.diff(div,axis=1)
    mask0 = df0 != 0
    mask1 = df1 != 0
    ret[0:height - 1, :, :] = np.logical_or(ret[0:height - 1, :, :], mask0)
    ret[1:height, :, :] = np.logical_or(ret[1:height, :,:], mask0)
    ret[:,  0:width-1, :] = np.logical_or(ret[:,  0:width-1,:],mask1)
    ret[:, 1:width, :] = np.logical_or(ret[:, 1:width,:], mask1)

    ret2 = np.ones([height,width,1], dtype="uint8")
    bounds = ret2 * 255 - ret * 255

    return bounds

def get_variable_name(variable):
    return [ k for k,v in locals().iteritems() if v == variable][0]

def save_to_folder(folder=None,filename=None, image=None, imType=None):
    if folder is None or filename is None or image is None:
        print("Could not save to file. {0}, {1} , {2} must not be None".format(folder, filename, image))
        return
    filename = folder +'/'+filename
    if len(image.shape) == 3:
        cv2.imwrite(filename, image)
    elif imType =="a":
        plt.imsave(filename, image)
    else:
       cv2.imwrite(filename, image)

    print("succesfully saved  to {0}".format(filename))

def reshape_image(image,size):
    img = np.reshape(image, (size[0], size[1], size[2])).astype(np.uint8)
    reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

    return reshaped, img  


def read_truth(path):
    """
    return the nparray of boundary (0 for boundary and 255 for area)
    :param path:
    :return:
    """
    mat = scipy.io.loadmat(path)
    ground_truth = mat.get('groundTruth')
    label_num = ground_truth.size

    for i in range(label_num):
        boundary = ground_truth[0][i]['Boundaries'][0][0]
        if i == 0:
            true_boundary = boundary
        else:
            true_boundary += boundary

    height = true_boundary.shape[0]
    width = true_boundary.shape[1]
    true_boundary = true_boundary.reshape(height, width, 1)

    true_boundary = 255 * np.ones([height, width, 1], dtype="uint8") - (true_boundary > 0) * 255

    return true_boundary

def helper(b1,b2,h,w,thres):
    cnt = 0
    for i in range(h):
        for j in range(w):
            if b1[i][j]:
                lower_x = max(0,i-thres)
                upper_x = min(h-1,i + thres)
                lower_y = max(0,j-thres)
                upper_y = min(w-1,j + thres)
                matrix_rows = b2[lower_x : upper_x + 1, :]
                matrix = matrix_rows[:, lower_y : upper_y+1]
                if matrix.sum() > 0:
                    cnt = cnt + 1
    total = b1.sum()
    return cnt/ total

    

def eval_bound(mask1, mask2 ,thres):
    '''Evaluate precision for boundary detection'''
    s1 = mask1.shape
    s2 = mask2.shape

    if s1 != s2:
        print ( 'shape not match')
        print ("swaping dimensions")
        s2 = s1
        
    if len(s1) == 3:
        b1 = mask1.reshape( s1[0], s1[1]) == 0
        b2 = mask2.reshape( s2[0], s2[1]) == 0
    else :
        b1 = mask1 == 0
        b2 = mask2 == 0
    
    h = s1[0]
    w = s1[1]
    precision = helper(b1,b2,h,w,thres)  
    recall = helper(b2,b1,h,w,thres)
    f1 = f1_score(precision, recall)
    # F1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def f1_score(precision, recall):
    return 2 *(precision * recall) / (precision + recall)