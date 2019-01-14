import cv2
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np
from time import time
from utils.tools import Tools


class Fuzzy:

    def segment(self, images, im_size, clusters=[3]):
        # looping every images
        for index,rgb_img in enumerate(images):
            img = np.reshape(rgb_img, (im_size[0],im_size[1], 3)).astype(np.uint8)
            shape = np.shape(img)
            # import pdb; pdb.set_trace()
            # initialize graph
            plt.figure(figsize=(20,20))
            plt.subplot(1,4,1)
            plt.imshow(img)
            plt.title("Original image")
            # looping every cluster     
            print('Image '+str(index+1))
            for i,cluster in enumerate(clusters):
                    
                # Fuzzy C Means
                new_time = time()
                
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                rgb_img.T, cluster, 2, error=0.005, maxiter=1000, init=None,seed=42)

                new_img =  Tools.change_color_fuzzycmeans(u,cntr)
                
                fuzzy_img = np.reshape(new_img,shape).astype(np.uint8)
                
                ret, seg_img = cv2.threshold(fuzzy_img,np.max(fuzzy_img)-1,255,cv2.THRESH_BINARY)
                
                print('Clustering (Fuzzy)',cluster)
                print(time() - new_time,'seconds')
                seg_img_1d = seg_img[:,:,1]
                
                
                bwfim1 =  Tools.bwareaopen(seg_img_1d, 100)
                bwfim2 =  Tools.imclearborder(bwfim1)
                bwfim3 =  Tools.imfill(bwfim2)
                
                print('Bwarea : '+str( Tools.bwarea(bwfim3)))

                plt.subplot(1,4,i+2)
                plt.imshow(bwfim3)
                name = str(cluster)+ ' Cluster (Fuzzy)'
                plt.title(name)

            name = 'segmented'+str(index)+'.png'
        plt.show()