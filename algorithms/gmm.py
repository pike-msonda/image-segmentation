from sklearn.mixture import GaussianMixture
from scipy.ndimage import median_filter
from skimage.morphology import label as cl
import matplotlib.pyplot as plt
from time import time
import numpy as np
import utils.calculate_boundary as cb
import utils.save_to_folder as stf

class GMM:
    def segment(self, images, im_size, filenames, clusters=[3]):
        
        print ("Segmenting using GMM")
        for index,(rgb_img, filename) in enumerate(zip(images, filenames)):
            img = np.reshape(rgb_img, (im_size[0], im_size[1], 3)).astype(np.uint8)
            reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
            
            new_time = time()
            print('Image '+str(index+1))

            for plotindex, cluster in enumerate(clusters):
                gmm = GaussianMixture(n_components=cluster, covariance_type="tied")
                gmm = gmm.fit(reshaped)
                gmm_clusters = gmm.predict(reshaped)
                clustering = np.reshape(np.array(gmm_clusters, dtype=np.uint8),
                    (img.shape[0], img.shape[1]))
                
                image_mask = cb.find_bound(clustering, im_size)
                # Sort the cluster labels in order of the frequency with which they occur.
                sortedLabels = sorted([n for n in range(cluster)],
                    key=lambda x: -np.sum(clustering == x))

                 # Initialize K-means grayscale image; set pixel colors based on clustering.
                gmmImage = np.zeros(img.shape[:2], dtype=np.uint8)
                
                for i, label in enumerate(sortedLabels):
                    gmmImage[clustering == label] = int(255 / (cluster - 1)) * i

                filename = "gmm/"+ filename
            stf.save_seg(gmmImage,filename)
            stf.save_to_binary(image_mask,filename)
    