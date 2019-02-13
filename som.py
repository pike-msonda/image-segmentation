import numpy as np
from minisom import MiniSom
from time import time
from scipy.ndimage import median_filter
from skimage.morphology import label as cl
import matplotlib.pyplot as plt
import utils.calculate_boundary as cb
import utils.save_to_folder as stf

class SOM:
    def segment(self, images, im_size, filenames, clusters=[3]):
        print ("Segmenting using SOM")
        for index,(rgb_img, filename) in enumerate(zip(images, filenames)):
            img = np.reshape(rgb_img, (im_size[0], im_size[1], 3)).astype(np.uint8)
            reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
            
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
                
                image_mask = cb.find_bound(clustering, im_size)
                 # Sort the cluster labels in order of the frequency with which they occur.
                sortedLabels = sorted([n for n in range(cluster)],
                    key=lambda x: -np.sum(clustering == x))

                 # Initialize K-means grayscale image; set pixel colors based on clustering.
                somImage = np.zeros(img.shape[:2], dtype=np.uint8)
                
                for i, label in enumerate(sortedLabels):
                    somImage[clustering == label] = int(255 / (cluster - 1)) * i 
            
                filename = "som/"+ filename
            stf.save_seg(somImage,filename)
            stf.save_to_binary(image_mask,filename)
                
        


