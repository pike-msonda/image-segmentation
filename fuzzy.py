import cv2
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np
from time import time
from utils.tools import Tools
from skimage.morphology import disk
from skimage.filters import gaussian, median
import utils.calculate_boundary as cb
import utils.save_to_folder as stf

class Fuzzy:

    """
        Image Segmentation with Fuzzy C means. 
    """
    def segment(self, images, im_size, filenames, clusters=[3]):
        # looping every images
        print ("Segmenting using FUZZY C MEANS")
        for index,(rgb_img, filename) in enumerate(zip(images, filenames)):
            img = np.reshape(rgb_img, (im_size[0],im_size[1], 3)).astype(np.uint8)
            shape = np.shape(img)

            print('Image '+str(index+1))
            for i,cluster in enumerate(clusters):
                filtered_image = median(rgb_img, disk(10))
                fuzzyImage, image_mask = self.fuzzy_seg(filtered_image, im_size, cluster)
                unfilteredImage, image_mask = self.fuzzy_seg(rgb_img, im_size, cluster)
                
                filename = "fuzzy/"+ filename
                unfiltered = "fuzzy/unfiltered/"+"unfiltered_"+filename

            stf.save_seg(fuzzyImage,filename)
            stf.save_seg(unfilteredImage,unfiltered)
            stf.save_to_binary(image_mask,filename)

    def fuzzy_seg(self, image,im_size,cluster):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    image.T, cluster, 2, error=0.005, maxiter=1000, init=None,seed=42)

        # make labels based on membership. 
        cluster_membership = np.argmax(u, axis=0)  
        
        # create binary image. Contains information about image boundary. 
        image_mask = cb.find_bound(cluster_membership, im_size)

        clustering = np.reshape(np.array(cluster_membership, dtype=np.uint8),
            (image.shape[0], image.shape[1]))
        
        sortedLabels = sorted([n for n in range(cluster)],
            key=lambda x: -np.sum(clustering == x))

        fuzzyImage = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, label in enumerate(sortedLabels):
            fuzzyImage[clustering == label] = int(255 / (cluster - 1)) * i

        print('Clustering (Fuzzy)',cluster)
        
        # get black and white area. 
        print('Bkack and White Area : '+str( Tools.bwarea(fuzzyImage)))

        # print(time() - new_time,'seconds')

        return fuzzyImage, image_mask