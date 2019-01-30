import numpy as np
from sklearn.cluster import KMeans
from time import time
from utils.tools import Tools
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import gaussian


class Kmeans:

    def segment(self, images, im_size, clusters=[3]):
        for index,rgb_img in enumerate(images):
            img = np.reshape(rgb_img, (im_size[0],im_size[1], 3)).astype(np.uint8)
            reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
            plt.figure(figsize=(20,20))
            plt.subplot(1,4,1)
            plt.imshow(img)
            plt.title("Original image")
            
            new_time = time()
            # looping every cluster  
            print('Image '+str(index+1))

            for plotindex,cluster in enumerate(clusters):
                filtered_image = gaussian(reshaped, sigma=1, multichannel=True)
                # import pdb; pdb.set_trace()
                kmeans = KMeans(n_clusters=cluster, n_init=40, max_iter=500).fit(filtered_image)
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

                # import pdb; pdb.set_trace() 
                
                print('Bkack and White Area : '+str(  Tools.bwarea(kmeansImage)))

                plt.subplot(1,4,plotindex+2)
                plt.imshow(kmeansImage)
                name = str(cluster)+ ' Cluster (KMEANS)'
                plt.title(name)

            name = 'segmented'+str(index)+'.png'
            # plt.savefig(name)
        plt.show()