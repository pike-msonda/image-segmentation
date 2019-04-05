from algorithms.segmentation_resolver import SegmentationResolver
from utils.utils import read_images
from datetime import datetime

CLUSTERS = [5]
IMAGES_FOLDER = "images"

def execute(images, size, img_names, method=None):
    params = {'images':images, 'image_size':size,'filenames':img_names, 
        'clusters':CLUSTERS, 'filter':'gaussian'}
    segmentation_algorithms = SegmentationResolver(params)
    
    segmentation_algorithms.segment(algorithm=method.lower())
        

def main():
    start = datetime.now()
    images, img_names, size =  read_images(folder=IMAGES_FOLDER)
    execute(images, size, img_names, method='KMEANS') 
    # execute(method='FUZZY') 
    # execute(method='MEANSHIFT')
    # execute(method='SOM')
    # execute(method='GMM')
    # execute(method='DBSCAN') # too slow. 
    time_elapsed = datetime.now() - start
    
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        
if __name__ =='__main__':
    main()
    #I have added new stuff.
    

