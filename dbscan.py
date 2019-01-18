from PIL import Image
import numpy as np
from sklearn.cluster import  DBSCAN, estimate_bandwidth
from time import time
import matplotlib.pyplot as plt
from utils.tools import Tools
from skimage.morphology import closing


class Dbscan:

	def segment(self, images, im_size, clusters=2):

		for index,rgb_img in enumerate(images):
			img = np.reshape(rgb_img, (im_size[0],im_size[1],3)).astype(np.uint8)

			flat_image = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

			dbscan = DBSCAN(eps=5, min_samples=50, metric="euclidean", 
				algorithm='auto').fit(flat_image)
			
			labels = dbscan.labels_
			labels_unique = np.unique(labels)
			n_clusters_ = len(labels_unique)
			n_noise_ = list(labels).count(-1)
			print("number of estimated clusters : %d" % n_clusters_)
			print("number of estimated noise : %d" % n_noise_)
			
			plt.figure(figsize=(20,20))
			plt.subplot(1,n_clusters_+1,1)
			plt.imshow(img)	
			plt.title("Original image")
			
			for k in range(n_clusters_):
			
				clustering = np.reshape(np.array(labels, dtype=np.uint8),
					(img.shape[0], img.shape[1]))

				sortedLabels = sorted([n for n in range(k + clusters)], 
					key=lambda x: -np.sum(clustering == x))
					
				dbscanImage = np.zeros(img.shape[:2], dtype=np.uint8)
			
				for i, label in enumerate(sortedLabels):
					dbscanImage[clustering == label] = int(255 / ((k + clusters) - 1)) * i

				plt.subplot(1,n_clusters_+1,k+2)
				plt.imshow(dbscanImage)
				plt.title("{0} Cluster (DBSCAN)".format(k+1))

		plt.show()

