from utils.utils import read_truth, eval_bound
import os
import cv2
import argparse
from tabulate import tabulate
import pandas as pd
from PIL import Image
import numpy as np

SEG_PATH = "./segs/"
BINARY_PATH = "./binary/"
GROUND_TRUTH ="./ground_truth/"

def main(args):
    truth_list = [f for f in os.listdir(GROUND_TRUTH) if os.path.isfile(os.path.join(GROUND_TRUTH, f)) and f.endswith(".mat")]
    seg_list = os.listdir(SEG_PATH)
    binary_list = os.listdir(BINARY_PATH)
    results = pd.DataFrame(columns=['Algorithm', 'Image', 'Precision', 'Recall', 'F1 Score'])

    if(args.d == 'D'):
        for dl in binary_list:
            seg_path = SEG_PATH + dl + '/'  
            binary_path = BINARY_PATH + dl + '/'
            for b in [f for f in os.listdir(binary_path) if os.path.isfile(os.path.join(binary_path, f)) and f.endswith(".png")]:
                if(os.path.exists(binary_path + b)):
                    print ("removing .. {0}".format(binary_path + b))
                    os.remove(binary_path + b)
            for s in  [f for f in os.listdir(seg_path) if os.path.isfile(os.path.join(seg_path, f)) and f.endswith((".png",".jpg"))]:
                if(os.path.exists(seg_path + s)):
                    print ("removing .. {0}".format(seg_path + s))
                    os.remove(seg_path + s)

        print("Delete was successful")
        return

    print("CALCULATING ...  \n")
    for sl in seg_list:
        path = SEG_PATH + sl + '/'
        if (sl == args.a or args.a == 'all'):
            for f, s in zip(truth_list, [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(".png")]):
                ground_path = GROUND_TRUTH + f
                boundary_path = path + s

                image_name = boundary_path.split('/')[-1]
                f_truth = read_truth(ground_path)
                img = cv2.imread(boundary_path)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                size = img.shape
                
                boundary_predict = img.reshape(size[0], size[1], 1)

                precision, recall, f1 = eval_bound(boundary_predict, f_truth, 10)
                data = {'Algorithm': sl.upper(), 'Image': image_name, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
                print ("boundary details: {0}, {1}".format(f_truth.shape[0], f_truth.shape[1]))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', default='all', type=str, required=False, help="select type of algorithm to evaluation. Eg.'kmeans', 'all', 'dbscan'")
    parser.add_argument('-d', type=str, required=False, help='deletes all segments and binary files')
    args = parser.parse_args()  
    main(args)