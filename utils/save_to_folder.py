import cv2;
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def save_seg(seg, filename, folder="segs"):
    filename = folder +'/'+filename
    cv2.imwrite(filename, seg)

def save_to_binary(binary, filename, folder="binary"):
     filename = folder +'/'+filename
     cv2.imwrite(filename, binary)

def read_truth(path):
    """
    return the nparray of boundary (0 for boundary and 255 for area)
    :param path:
    :return:
    """
    mat = scipy.io.loadmat(path)
    groundTruth = mat.get('groundTruth')
    label_num = groundTruth.size

    for i in range(label_num):
        boundary = groundTruth[0][i]['Boundaries'][0][0]
        if i == 0:
            trueBoundary = boundary
        else:
            trueBoundary += boundary

    height = trueBoundary.shape[0]
    width = trueBoundary.shape[1]
    trueBoundary = trueBoundary.reshape(height, width, 1)

    trueBoundary = 255 * np.ones([height, width, 1], dtype="uint8") - (trueBoundary > 0) * 255

    return trueBoundary