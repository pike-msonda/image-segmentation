import numpy as np
from sklearn.cluster import KMeans
from time import time
from utils.tools import Tools
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import gaussian, median
import utils.calculate_boundary as cb
import utils.save_to_folder as stf

class Kmeans:
    """
        Params:
            images: array. List of images to segment
            im_size: tuple. The size of the images to be segmented. 
            filnames: array of strings. List of filenames of the images to be segmented. 
            filters: string. Default is "MEDIAN" filter. Supports GAUSSIAN.  
    """
    def __init__(self, images, im_size, filenames,clusters, filters="median"):
        self.images = images
        self.im_size = im_size
        self.filenames = filenames
        self.clusters = clusters
        self.filters = filters

    """
        Segmentation method. Applies filters to the image. 
        Segmentation is done by K-MEANS and saved to output folders.
        returns: None

    """
    def segment(self):
        print ("Segmenting using K-MEANS")
        for index,(rgb_img, filename) in enumerate(zip(self.images, self.filenames)):
            img = np.reshape(rgb_img, (self.im_size[0],self.im_size[1], 3)).astype(np.uint8)
            reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])  

            print('Segmenting Image '+str(index+1))

            for index, cluster in enumerate(self.clusters):
                if self.filters.upper() == "MEDIAN":
                    print("Using {0} filter....".format(self.filters.upper()))
                    filtered_image = median(reshaped, disk(10))
                elif self.filters.upper() == "GAUSSIAN":
                    print("really works")
                    print("Using {0} filter....".format(self.filters.upper()))
                    filtered_image = gaussian(reshaped, sigma=1, multichannel=True)

                kmeansImage, image_mask = self.kmeans(filtered_image, img, self.im_size, cluster)
                unfilteredImage, unfiltered_image_mask =  self.kmeans(reshaped, img, self.im_size, cluster)

            unfiltered = "kmeans/unfiltered/"+"unfiltered_"+filename
            filename = "kmeans/"+ filename

            stf.save_seg(kmeansImage,filename)
            stf.save_seg(unfilteredImage,unfiltered)
            stf.save_to_binary(image_mask,filename)
    """
        K-MEANS algorithm to segment image. 
        params:
            image:  array
            img:    array
            im_size: tuple
            cluster: array
        returns: 
            kmeansImage: array: segmented image
            image_mask: array: boundary of the image. 

    """
    def kmeans(self, image, img, im_size, cluster):
        kmeans = KMeans(n_clusters=cluster, n_init=40, max_iter=500).fit(image)
        image_mask = cb.find_bound(kmeans.labels_, im_size)
        clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
            (img.shape[0], img.shape[1]))
        
        sortedLabels = sorted([n for n in range(cluster)],
            key=lambda x: -np.sum(clustering == x))
        
        kmeansImage = np.zeros(img.shape[:2], dtype=np.uint8)
        for i, label in enumerate(sortedLabels):
            kmeansImage[clustering == label] = int(255 / (cluster - 1)) * i

        return kmeansImage, image_mask, 