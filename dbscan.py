from PIL import Image
import numpy as np
from sklearn.cluster import  DBSCAN, estimate_bandwidth
from time import time
import matplotlib.pyplot as plt
from utils.tools import Tools
from skimage.morphology import closing
import utils.calculate_boundary as cb
import utils.save_to_folder as stf

class Dbscan:

	def segment(self, images, im_size, filenames, clusters=3):

		for index,(rgb_img, filename) in enumerate(zip(images, filenames)):
			img = np.reshape(rgb_img, (im_size[0],im_size[1],3)).astype(np.uint8)

			flat_image = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

			dbscan = DBSCAN(eps=5, min_samples=50, metric="euclidean", 
				algorithm='auto').fit(flat_image)
			
			labels = dbscan.labels_
			image_mask = cb.find_bound(labels, im_size)
			labels_unique = np.unique(labels)
			n_clusters_ = len(labels_unique)
			n_noise_ = list(labels).count(-1)
			print("number of estimated clusters : %d" % n_clusters_)
			print("number of estimated noise : %d" % n_noise_)
			
			n_clusters_ = range(2, n_clusters_ +1)

			clustering = np.reshape(np.array(labels, dtype=np.uint8),
				(img.shape[0], img.shape[1]))

			sortedLabels = sorted([n for n in range(clusters)], 
				key=lambda x: -np.sum(clustering == x))
				
			dbscanImage = np.zeros(img.shape[:2], dtype=np.uint8)
		
			for i, label in enumerate(sortedLabels):
				dbscanImage[clustering == label] = int(255 / (clusters- 1)) * i
			
				
			filename = "dbscan/"+ filename

			stf.save_seg(dbscanImage,filename)
			stf.save_to_binary(image_mask,filename)
                

