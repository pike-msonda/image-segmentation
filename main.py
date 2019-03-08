from algorithms.segmentation_resolver import SegmentationResolver
from utils.utils import read_images
from datetime import datetime

CLUSTERS = [5]
IMAGES_FOLDER = "images"

def execute(method=None):
    images, img_names, size =  read_images(folder=IMAGES_FOLDER)

    params = {'images':images, 'image_size':size,'filenames':img_names, 
        'clusters':CLUSTERS, 'filter':'median'}
    segmentation_algorithms = SegmentationResolver(params)
    
    segmentation_algorithms.segment(algorithm=method.lower())
        

def main():
    start = datetime.now()
    execute(method='KMEANS')
    execute(method='FUZZY') 
    execute(method='MEANSHIFT')
    execute(method='SOM')
    execute(method='GMM')
    # execute(method='DBSCAN') # too slow. 
    time_elapsed = datetime.now() - start
    
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        
if __name__ =='__main__':
   main()

