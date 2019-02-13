import utils.save_to_folder as stf
from os import listdir, remove, path
import cv2
import argparse
from os.path import isfile, join, splitext
import utils.evaluate_boundary as eb
from tabulate import tabulate
import pandas as pd

SEG_PATH = "./segs/"
BINARY_PATH = "./binary/"
GROUND_TRUTH ="./ground_truth/"

def main(args):
    truth_list = [f for f in listdir(GROUND_TRUTH) if isfile(join(GROUND_TRUTH, f)) and f.endswith(".mat")]
    seg_list = listdir(SEG_PATH)
    binary_list = listdir(BINARY_PATH)
    results = pd.DataFrame(columns=['Algorithm', 'Image', 'Precision', 'Recall'])
    
    print("CALCULATING ...  \n")
    for sl in seg_list:
        path = SEG_PATH + sl + '/'
        if (sl == args.a or args.a == 'all'):
            for f, s in zip(truth_list, [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".jpg")]):
                ground_path = GROUND_TRUTH + f
                boundary_path = path + s

                image_name = boundary_path.split('/')[-1]
                f_truth = stf.read_truth(ground_path)
                img = cv2.imread(boundary_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                size = img.shape

        
                boundary_predict = img.reshape(size[0], size[1], 1)

                precision, recall = eb.eval_bound(boundary_predict, f_truth, 2)
                
                data = {'Algorithm': sl.upper(), 'Image': image_name, 'Precision': precision, 'Recall': recall}
                print ("boundar details: {0}, {1}".format(f_truth.shape[0], f_truth.shape[1]))
                print ("image details: {0}, {1}".format(img.shape[0], img.shape[1]))

                array = []
                for key, value in data.items():
                        temp = [key,value]
                        array.append(temp)
                print("appending result using {0} and {1}".format(ground_path, boundary_path))

                print(tabulate(array))
                print("")

                results = results.append(data, ignore_index=True)

    print("Average results")
    print("======================================")
    print (results.groupby(['Algorithm']).mean())

    if(args.d == 'D'):
        for dl in binary_list:
            seg_path = SEG_PATH + dl + '/'  
            binary_path = BINARY_PATH + dl + '/'  
            for f, s in zip([f for f in listdir(binary_path) if isfile(join(binary_path, f)) and f.endswith(".jpg")], 
                [f for f in listdir(seg_path) if isfile(join(seg_path, f)) and f.endswith(".jpg")]):
                if(path.exists(binary_path + f)):
                    print ("removing .. {0}".format(binary_path + f))
                    remove(binary_path + f)
                if(path.exists(seg_path + s)):
                    print ("removing .. {0}".format(binary_path + f))
                    remove(seg_path + f)
        print("completed")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', default='all', type=str, required=False, help="select type of algorithm to evaluation. Eg.'kmeans', 'all', 'dbscan'")
    parser.add_argument('-d', type=str, required=False, help='deletes all segments and binary files')
    args = parser.parse_args()  
    # import pdb; pdb.set_trace()
    main(args)