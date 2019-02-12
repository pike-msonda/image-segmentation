from PIL import Image
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from time import time
import matplotlib.pyplot as plt
from utils.tools import Tools
from skimage.morphology import closing
import utils.calculate_boundary as cb
import utils.save_to_folder as stf

class Mean:
     def segment(self, images, im_size, filenames, clusters=[3]):
        for index,(rgb_img, filename) in enumerate(zip(images, filenames)):
            img = np.reshape(rgb_img, (im_size[0],im_size[1],3)).astype(np.uint8)
            flat_image = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        
            bandwidth = estimate_bandwidth(flat_image, quantile=.2, n_samples=100)
            print("The bandwith of image {0} is {1}".format(index, bandwidth))

            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq = 100)
            ms.fit(flat_image)

            image_mask = cb.find_bound(ms.labels_, im_size)

            labels = ms.labels_
            cluster_centers =  np.uint8(ms.cluster_centers_)
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            print("number of estimated clusters : %d" % n_clusters_)
            center_results = cluster_centers[labels.flatten(), 0:3]
            meanShiftImage = center_results.reshape(rgb_img.shape)
            n_clusters_ = range(2, n_clusters_ + 1)
            
            for plotindex, cluster in enumerate(n_clusters_):
                clustering = np.reshape(np.array(labels, dtype=np.uint8),
                    (img.shape[0], img.shape[1]))
                
                sortedLabels = sorted([n for n in range(cluster)],
                    key=lambda x: -np.sum(clustering == x))

                meanShiftImage = np.zeros(img.shape[:2], dtype=np.uint8)

                for i, label in enumerate(sortedLabels):
                    meanShiftImage[clustering == label] = int(255 / (cluster - 1)) * i

            filename = "meanshift/"+ filename
            
            stf.save_seg(meanShiftImage,filename)
            stf.save_to_binary(image_mask,filename)
