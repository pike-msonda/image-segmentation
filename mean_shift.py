from PIL import Image
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from time import time
import matplotlib.pyplot as plt
from utils.tools import Tools
from skimage.morphology import closing

class Mean:
    def segment(self, images, im_size, clusters=3):
        for index,rgb_img in enumerate(images):
            img = np.reshape(rgb_img, (im_size[0],im_size[1],3)).astype(np.uint8)
            flat_image = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        
            bandwidth = estimate_bandwidth(flat_image, quantile=.2, n_samples=100)
            print("The bandwith of image {0} is {1}".format(index, bandwidth))

            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq = 100)
            ms.fit(flat_image)

            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            print("number of estimated clusters : %d" % n_clusters_)

            plt.figure(figsize=(20,20))
            plt.subplot(1,n_clusters_+1,1)
            plt.imshow(img)
            plt.title("Original image")

            for k in range(n_clusters_):
                
                clustering = np.reshape(np.array(labels, dtype=np.uint8),
                    (img.shape[0], img.shape[1]))

                sortedLabels = sorted([n for n in range(k+clusters)],
                    key=lambda x: -np.sum(clustering == x))

                meanShiftImage = np.zeros(img.shape[:2], dtype=np.uint8)

                for i, label in enumerate(sortedLabels):
                    meanShiftImage[clustering == label] = int(255 / ((k+clusters) - 1)) * i

                plt.subplot(1,n_clusters_+1,k+2)
                plt.imshow(meanShiftImage)
                plt.title("{0} Cluster (MEAN-SHIFT)".format(k+2))

        plt.show()
