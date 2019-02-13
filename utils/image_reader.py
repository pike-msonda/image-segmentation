import os
import cv2

class ImageReader:

    def __init__(self, folder=None):
        self.folder = folder
        # self.width = width
        # self.height = height

    def read(self):
        folder = self.folder + "/"
        print("Reading images from {0}".format(folder))
        list_images = os.listdir(folder)
        list_img = []
        list_img_name = []
        for i in list_images:
            path = folder+i
            print(path)
            list_img_name.append(path.split('/')[1])
            img = cv2.imread(path)
            print(img.shape)
            self.width = img.shape[0]
            self.height = img.shape[1]
            # import pdb; pdb.set_trace()
            # if self.width is not None and self.height is not None:
            #     img = cv2.resize(img, (self.width, self.height),interpolation=cv2.INTER_AREA)
            rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
            list_img.append(rgb_img)
        return list_img, list_img_name

    def size(self):
            return self.width, self.height