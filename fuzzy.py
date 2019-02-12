import cv2
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np
from time import time
from utils.tools import Tools
import utils.calculate_boundary as cb
import utils.save_to_folder as stf

class Fuzzy:

    def segment(self, images, im_size, filenames, clusters=[3]):
        # looping every images
        for index,(rgb_img, filename) in enumerate(zip(images, filenames)):
            img = np.reshape(rgb_img, (im_size[0],im_size[1], 3)).astype(np.uint8)
            shape = np.shape(img)

            # looping every cluster     
            print('Image '+str(index+1))
            for i,cluster in enumerate(clusters):
                    
                # Fuzzy C Means
                new_time = time()
                
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    rgb_img.T, cluster, 2, error=0.005, maxiter=1000, init=None,seed=42)

                # import pdb; pdb.set_trace()
                new_img =  Tools.change_color_fuzzycmeans(u,cntr)
                
                fuzzy_img = np.reshape(new_img,shape).astype(np.uint8)

                image_mask = cb.find_bound(fuzzy_img[:,:,2], im_size)
                # ret, seg_img = cv2.threshold(fuzzy_img,np.max(fuzzy_img)-1,255,cv2.THRESH_BINARY)
                
                print('Clustering (Fuzzy)',cluster)
                print(time() - new_time,'seconds')
                seg_img_1d = fuzzy_img[:,:,1]
                
                
                # bwfim1 =  Tools.bwareaopen(seg_img_1d, 100)
                # bwfim2 =  Tools.imclearborder(bwfim1)
                # bwfim3 =  Tools.imfill(bwfim2)
                
                print('Bwarea : '+str( Tools.bwarea(seg_img_1d)))

                # plt.subplot(1,4,i+2)
                # plt.imshow(seg_img_1d)
                # name = str(cluster)+ ' Cluster (Fuzzy)'
                # plt.title(name)
                filename = "fuzzy/"+ filename
            stf.save_seg(seg_img_1d,filename)
            stf.save_to_binary(image_mask,filename)
            # name = 'segmented'+str(index)+'.png'
        # plt.show()