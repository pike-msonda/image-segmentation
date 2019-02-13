import numpy as np
from sklearn.cluster import KMeans
from time import time
from utils.tools import Tools
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import gaussian
import utils.calculate_boundary as cb
import utils.save_to_folder as stf

class Kmeans:

    def segment(self, images, im_size, filenames, clusters=[3]):
        
        print ("Segmenting using K-MEANS")
        for index,(rgb_img, filename) in enumerate(zip(images, filenames)):
            img = np.reshape(rgb_img, (im_size[0],im_size[1], 3)).astype(np.uint8)
            reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

            new_time = time()
            # looping every cluster  
            print('Image '+str(index+1))

            for plotindex,cluster in enumerate(clusters):
                filtered_image = gaussian(reshaped, sigma=1, multichannel=True)
                kmeans = KMeans(n_clusters=cluster, n_init=40, max_iter=500).fit(filtered_image)

                # calculate the boundary of an image. 
                image_mask = cb.find_bound(kmeans.labels_, im_size)
                # import pdb; pdb.set_trace()
                clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
                    (img.shape[0], img.shape[1]))
               
                # Sort the cluster labels in order of the frequency with which they occur.
                sortedLabels = sorted([n for n in range(cluster)],
                    key=lambda x: -np.sum(clustering == x))
                    
                # Initialize K-means grayscale image; set pixel colors based on clustering.
                kmeansImage = np.zeros(img.shape[:2], dtype=np.uint8)
                for i, label in enumerate(sortedLabels):
                    kmeansImage[clustering == label] = int(255 / (cluster - 1)) * i

                print('Clustering (Kmeans)',cluster)
                print(time() - new_time,'seconds')


                # get black and white araa.     
                print('Bkack and White Area : '+str(  Tools.bwarea(kmeansImage)))

                filename = "kmeans/"+ filename

            stf.save_seg(kmeansImage,filename)
            stf.save_to_binary(image_mask,filename)
    
  