from sklearn.mixture import GaussianMixture
from scipy.ndimage import median_filter
from skimage.morphology import label as cl
import matplotlib.pyplot as plt
from time import time
import numpy as np

class GMM:
    def segment(self, images, im_size, clusters=[3]):
        for index, rgb_img in enumerate(images):
            img = np.reshape(rgb_img, (im_size[0], im_size[1], 3)).astype(np.uint8)
            reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
            
            plt.figure(figsize=(20,20))
            plt.subplot(1,4,1)
            plt.imshow(img)
            plt.title("Original image")
             
            new_time = time()
            # looping every cluster  
            print('Image '+str(index+1))

            for plotindex, cluster in enumerate(clusters):
                gmm = GaussianMixture(n_components=cluster, covariance_type="tied")
                gmm = gmm.fit(reshaped)
                gmm_clusters = gmm.predict(reshaped)
                clustering = np.reshape(np.array(gmm_clusters, dtype=np.uint8),
                    (img.shape[0], img.shape[1]))
                
                # Sort the cluster labels in order of the frequency with which they occur.
                sortedLabels = sorted([n for n in range(cluster)],
                    key=lambda x: -np.sum(clustering == x))

                 # Initialize K-means grayscale image; set pixel colors based on clustering.
                gmmImage = np.zeros(img.shape[:2], dtype=np.uint8)
                
                for i, label in enumerate(sortedLabels):
                    gmmImage[clustering == label] = int(255 / (cluster - 1)) * i
                # gmm_clusters = gmm_clusters.reshape(im_size[0], im_size[1])
                # # calculate connected components
                # cc_image = cl(gmm_clusters, connectivity=2)
                # # add filters
                # clusters_filtered = median_filter(clusters, 7)
                # clusters_filtered, cc_image
                
                plt.subplot(1,4,plotindex+2)
                plt.imshow(gmmImage)
                name = str(cluster)+ ' Cluster (GMM)'
                plt.title(name)
        plt.show()