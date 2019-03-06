from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from minisom import MiniSom
import skfuzzy as fuzz
from utils.utils import *

class SegmentationResolver:
    """
        Params:
            images: array. List of images to segment
            im_size: tuple. The size of the images to be segmented. 
            filnames: array of strings. List of filenames of the images to be segmented. 
            filters: string. Default is "MEDIAN" filter. Supports GAUSSIAN.  
    """
    def __init__(self, params):
        self.images = params['images']
        self.im_size = params['image_size']
        self.filenames = params['filenames']
        self.clusters = params['clusters']
        self.filter = [params['filter'], 'median'][params['filter'] == None] 
        self.dispatcher = { 'kmeans' : self.kmeans, 'fuzzy':self.fuzzy, 
            'dbscan': self.dbscan, 'som':self.som, 'meanshift':self.mean_shift, 'gmm': self.gmm}
    """
        Segmentation method. Applies filters to the image. 
        Segmentation is done by K-MEANS and saved to output folders.
        returns: None

    """
    def segment(self, algorithm="kmeans"):
        print ("Segmenting using {0}.".format(algorithm.upper()))
        for index,(rgb_img, filename) in enumerate(zip(self.images, self.filenames)):
            # transform image array into a 2-D array. 
            reshaped, image, = reshape_image(rgb_img, self.im_size)

            print('Segmenting Image '+str(index+1))
            
            for i,cluster in enumerate(self.clusters):
                 # filter image 
                filtered_image = filter_image(self.filter, reshaped)

                segmented_image, image_boundaries = self.execute_algorithm(algorithm,filtered_image,image, self.im_size, cluster)
                unfiltered_segment, unfiltered_boundaries = self.execute_algorithm(algorithm,reshaped, image, self.im_size, cluster)
                
                filec = algorithm +"/" +filename.split('.')[0] +'_clr.jpg'
                filename = filename.split('.')[0] + '.png' #using PNG to avoid dimension change when saving to file
                unfiltered = "/unfiltered/"+"unfiltered_"+filename
                filename = algorithm + "/"+ filename

               # save to a folder
                save_to_folder(folder='segs', filename=filename, image=segmented_image)
                save_to_folder(folder='segs', filename=filec, image=segmented_image, imType='a')
                save_to_folder(folder='binary', filename=filename, image=image_boundaries)
                save_to_folder(folder='segs/'+algorithm, filename=unfiltered, image=unfiltered_segment)
    
    def execute_algorithm(self,algorithm, image_to_segment, original_image, im_size, cluster):
        
        return self.dispatcher[algorithm](image_to_segment, original_image, im_size,cluster)
    

    def kmeans(self, image_to_segment, original_image, im_size, cluster):
        # clustering
        kmeans = KMeans(n_clusters=cluster, n_init=40, max_iter=500).fit(image_to_segment)
        # segmented image after grouping clustering labels
        kmeans_image, bounded_image = clustering(kmeans.labels_, original_image, cluster)
        return kmeans_image, bounded_image,

    def fuzzy(self, image_to_segment, original_image, im_size, cluster):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    image_to_segment.T, cluster, 2, error=0.005, maxiter=1000, init=None,seed=42)

        # make labels based on membership. 
        cluster_membership = np.argmax(u, axis=0)  
        
        fuzzy_image, bounded_image = clustering(cluster_membership, original_image, cluster)
        # create binary image. Contains information about image boundary. 

        return fuzzy_image, bounded_image

    def dbscan(self, image_to_segment, original_image, im_size, cluster):
        dbscan = DBSCAN(eps=5, min_samples=50, metric="euclidean", 
            algorithm='auto').fit(image_to_segment)
        
        dbscan_image, bounded_image = clustering( dbscan.labels_, original_image, cluster)
        # create binary image. Contains information about image boundary. 

        return dbscan_image, bounded_image

    def som(self, image_to_segment, original_image, im_size, cluster):
        som = MiniSom(cluster, 1, 3, sigma=0.1, learning_rate=0.5)              
        som.random_weights_init(image_to_segment)
        som.train_random(image_to_segment, 100)
        qnt = som.quantization(image_to_segment)
        z = som.get_weights().reshape(cluster, 3)
        z = np.sum(z, axis=1)
        z = z.tolist()
        labels = []
        for i, x in enumerate(qnt):
            labels += [z.index(np.sum(x))]
    
        labels = np.array(labels)
        som_image, bounded_image = clustering(labels, original_image, cluster)
        # create binary image. Contains information about image boundary. 

        return som_image, bounded_image

    def mean_shift(self, image_to_segment, original_image, im_size, cluster):

        bandwidth = estimate_bandwidth(image_to_segment, quantile=.2, n_samples=1000)
        print("The bandwith of image {0}".format(bandwidth))

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq = 100)
        ms.fit(image_to_segment)
        mean_image, bounded_image = clustering(ms.labels_, original_image, cluster)
        # create binary image. Contains information about image boundary. 

        return mean_image, bounded_image
    
    def gmm(self, image_to_segment, original_image, im_size, cluster):
        gmm = GaussianMixture(n_components=cluster, covariance_type="tied")
        gmm = gmm.fit(image_to_segment)
        labels = gmm.predict(image_to_segment)
        
        gmm_image, bounded_image = clustering(labels, original_image, cluster)
        # create binary image. Contains information about image boundary. 

        return gmm_image, bounded_image
        
        
