from utils.image_reader import ImageReader
from fuzzy import Fuzzy
from k_means import Kmeans
from datetime import datetime
from mean_shift  import Mean
from som import SOM
from gmm import GMM

CLUSTERS = [2,3,4]
def main(method=None,folder=None,width=None,height=None):

    if method is None:
        print('Specify segmentation method as main(method="KMEANS" or "FUZZY"')
        return
    else:
        image_reader = ImageReader(folder=folder,width=width, height=height)
        images =  image_reader.read()
        if method == "KMEANS":
            kmeans = Kmeans()
            print("Image size of: {0}".format(image_reader.size()))
            kmeans.segment(images=images, im_size=image_reader.size(), clusters=CLUSTERS)
        elif method =="FUZZY":
            fuzzy =  Fuzzy()
            fuzzy.segment(images=images,im_size=image_reader.size(),clusters=CLUSTERS)
        elif method == "MEAN":
            mean_shift = Mean()
            mean_shift.segment(images=images, im_size=image_reader.size(), clusters=2)
        elif method == "SOM":
            som = SOM()
            som.segment(images=images, im_size=image_reader.size(), clusters=CLUSTERS)
        elif method == "GMM":
            gmm = GMM()
            gmm.segment(images=images, im_size=image_reader.size(), clusters=CLUSTERS)
        else:
            print ("Not supported: {0}".format(method))
        

        
if __name__ =='__main__':
    start = datetime.now()
    # main(method='KMEANS', folder='images', width=200, height=200)
    # main(method='FUZZY', folder='images', width=200, height=200)
    # main(method='MEAN', folder='images', width=200, height=200)
    # main(method='SOM', folder='images', width=200, height=200)
    main(method='GMM', folder='images', width=200, height=200)

    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))