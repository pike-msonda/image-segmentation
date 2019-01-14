import numpy as np
from minisom import MiniSom
from time import time
from scipy.ndimage import median_filter
from skimage.morphology import label as cl
import matplotlib.pyplot as plt


class SOM:
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
                img_float = np.float32(reshaped)
                som = MiniSom(cluster, 1, 3, sigma=0.1, learning_rate=0.5)              
                som.random_weights_init(img_float)
                som.train_random(img_float, 100)
                qnt = som.quantization(img_float)
                z = som.get_weights().reshape(cluster, 3)
                z = np.sum(z, axis=1)
                z = z.tolist()
                output = []

                for i, x in enumerate(qnt):
                    output += [z.index(np.sum(x))]
            
                output = np.array(output)
                clustering = np.reshape(np.array(output, dtype=np.uint8),
                    (img.shape[0], img.shape[1]))
                
                 # Sort the cluster labels in order of the frequency with which they occur.
                sortedLabels = sorted([n for n in range(cluster)],
                    key=lambda x: -np.sum(clustering == x))

                 # Initialize K-means grayscale image; set pixel colors based on clustering.
                somImage = np.zeros(img.shape[:2], dtype=np.uint8)
                
                for i, label in enumerate(sortedLabels):
                    somImage[clustering == label] = int(255 / (cluster - 1)) * i 
                # output = output.reshape(im_size[0], im_size[1])
                # cc_image = cl(output, connectivity=2)
                # labels_filtered = median_filter(output,7)
                # labels_filtered, cc_image

                plt.subplot(1,4,plotindex+2)
                plt.imshow(somImage)
                name = str(cluster)+ ' Cluster (SOM)'
                plt.title(name)

        plt.show()


        


