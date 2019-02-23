import numpy as np
from sklearn.cluster import KMeans
from time import time
from utils.utils import *

class Kmeans:
    """
        Params:
            images: array. List of images to segment
            im_size: tuple. The size of the images to be segmented. 
            filnames: array of strings. List of filenames of the images to be segmented. 
            filters: string. Default is "MEDIAN" filter. Supports GAUSSIAN.  
    """
    def __init__(self, params):
        self.images = params['images']
        self.im_size = params['image_size']
        self.filenames = params['filenames']
        self.clusters = params['clusters']
        self.filter = [params['filter'], 'median'][params['filter'] == None] 

    """
        Segmentation method. Applies filters to the image. 
        Segmentation is done by K-MEANS and saved to output folders.
        returns: None

    """
    def segment(self):

        print ("Segmenting using K-MEANS")

        for index,(rgb_img, filename) in enumerate(zip(self.images, self.filenames)):

            reshaped, image, = reshape_image(rgb_img, self.im_size)

            print('Segmenting Image '+str(index+1))

            for index, cluster in enumerate(self.clusters):

                filtered_image = filter_image(self.filter, reshaped)

                segmented_image, image_boundaries = self.kmeans(filtered_image, 
                    image, self.im_size, cluster)

                unfiltered_segment, unfiltered_boundaries = self.kmeans(reshaped, 
                    image, self.im_size, cluster)

            unfiltered = "unfiltered/"+"unfiltered_"+filename
            filename = "kmeans/"+ filename

            save_to_folder(folder='segs', filename=filename, image=segmented_image)
            save_to_folder(folder='binary', filename=filename, image=image_boundaries)

            save_to_folder(folder='segs/kmeans', filename=unfiltered, image=unfiltered_segment)


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
    def kmeans(self, image_to_segment, original_image, im_size, cluster):
        kmeans = KMeans(n_clusters=cluster, n_init=40, max_iter=500).fit(image_to_segment)
        kmeans_image, bounded_image = clustering(kmeans.labels_, original_image, cluster)
        return kmeans_image, bounded_image, 