from algorithms.segmentation_resolver import SegmentationResolver
from utils.utils import read_images
from datetime import datetime

CLUSTERS = [3]
IMAGES_FOLDER = "images"

def execute(method=None):
    images, img_names, size =  read_images(folder=IMAGES_FOLDER)

    params = {'images':images, 'image_size':size,'filenames':img_names, 
        'clusters':CLUSTERS, 'filter':'gaussian'}
    segmentation_algorithms = SegmentationResolver(params)

    if method == "KMEANS":
        segmentation_algorithms.segment()
    elif method =="FUZZY":
         segmentation_algorithms.segment(algorithm=method.lower())
    # elif method == "MEAN":
    #     mean_shift = Mean()
    #     mean_shift.segment(images=images, im_size=image_reader.size(), filenames=img_names)
    # elif method == "SOM":
    #     som = SOM()
    #     som.segment(images=images, im_size=image_reader.size(), filenames=img_names, clusters=CLUSTERS)
    # elif method == "GMM":
    #     gmm = GMM()
    #     gmm.segment(images=images, im_size=image_reader.size(),filenames=img_names, clusters=CLUSTERS)
    # elif method == "DBSCAN":
    #     dbscan = Dbscan()
    #     dbscan.segment(images=images, im_size=image_reader.size(),filenames=img_names)
    # else:
    #     print ("Not supported: {0}".format(method))
        

def main():
    start = datetime.now()
    execute(method='KMEANS')
    execute(method='FUZZY')
    # execute(method='MEAN', folder='images')
    # execute(method='SOM', folder='images')
    # execute(method='GMM', folder='images')
    # execute(method='DBSCAN', folder='images') # too slow. 

    time_elapsed = datetime.now() - start
    
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        
if __name__ =='__main__':
   main()
