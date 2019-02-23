"""
    CRUNCH all the utility functions in here
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import gaussian, median
"""
=============================================================
"""
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
    bounds = ret2*255 - ret * 255

    return bounds

def get_variable_name(variable):
    return [ k for k,v in locals().iteritems() if v == variable][0]

def save_to_folder(folder=None,filename=None, image=None):
    if folder is None or filename is None or image is None:
        print("Could not save to file. {0}, {1} , {2} must not be None".format(folder, filename, image))
        return
    filename = folder +'/'+filename
    if len(image.shape) == 3:
       cv2.imwrite(filename, image)
    else:
        plt.imsave(filename, image)

    print("succesfully saved  to {0}".format(filename))

def reshape_image(image,size):
    img = np.reshape(image, (size[0],size[1], 3)).astype(np.uint8)
    reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

    return reshaped, image  