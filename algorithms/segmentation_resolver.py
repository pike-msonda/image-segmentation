from sklearn.cluster import KMeans
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
        self.dispatcher = { 'kmeans' : self.kmeans, 'fuzzy':self.fuzzy}
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
                
                unfiltered = "/unfiltered/"+"unfiltered_"+filename
                filename = algorithm + "/"+ filename

               # save to a folder
                save_to_folder(folder='segs', filename=filename, image=segmented_image)
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
