import os
import cv2
from os import listdir
from os.path import isfile, join, splitext 
class ImageReader:

    def __init__(self, folder=None):
        self.folder = folder

    def read(self):
        folder = self.folder + "/"
        print("Reading images from {0}".format(folder))
        list_images = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(".jpg")]
        list_img = []
        list_img_name = []
        for i in list_images:
            path = folder+i
            print(path)
            list_img_name.append(path.split('/')[1])
            img = cv2.imread(path)
            self.width = img.shape[0]
            self.height = img.shape[1]
            rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
            list_img.append(rgb_img)
        return list_img, list_img_name

    def size(self):
            return self.width, self.height