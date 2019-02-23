from utils.image_reader import ImageReader
from algorithms.k_means import Kmeans
from datetime import datetime

CLUSTERS = [3]
def main(method=None,folder=None):

    image_reader = ImageReader(folder=folder)
    
    images, img_names =  image_reader.read()

    params = {'images':images, 'image_size':image_reader.size(),
        'filenames':img_names, 'clusters':CLUSTERS, 'filter':'gaussian'}

    if method == "KMEANS":
        kmeans = Kmeans(params)
        kmeans.segment()
    # elif method =="FUZZY":
    #     fuzzy =  Fuzzy(params)
    #     fuzzy.segment(images=images,im_size=image_reader.size(), filenames=img_names, clusters=CLUSTERS)
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
        

        
if __name__ =='__main__':
    start = datetime.now()
    main(method='KMEANS', folder='images')
    # main(method='FUZZY', folder='images')
    # main(method='MEAN', folder='images')
    # main(method='SOM', folder='images')
    # main(method='GMM', folder='images')
    # main(method='DBSCAN', folder='images') # too slow. 

    time_elapsed = datetime.now() - start
    
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))