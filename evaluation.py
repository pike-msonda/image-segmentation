import utils.save_to_folder as stf
from os import listdir
import cv2
from os.path import isfile, join, splitext
import utils.evaluate_boundary as eb
from tabulate import tabulate
import pandas as pd

SEG_PATH = "./segs/"
BINARY_PATH = "./binary/"
GROUND_TRUTH ="./ground_truth/"

def main():
        truth_list = [f for f in listdir(GROUND_TRUTH) if isfile(join(GROUND_TRUTH, f)) and f.endswith(".mat")]
        seg_list = listdir(SEG_PATH)
        results = pd.DataFrame(columns=['Algorithm', 'Image', 'Precision', 'Recall'])
        
        print("CALCULATING ...  \n")
        
        for sl in seg_list:
                path = SEG_PATH + sl + '/'
            
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
                        
                        array = []
                        for key, value in data.items():
                               temp = [key,value]
                               array.append(temp)
                        print("Appending...")

                        print(tabulate(array))
                        print("")

                        results = results.append(data, ignore_index=True)
        
        print("Average results")
        print("======================================")
        print (results.groupby(['Algorithm']).mean())

if __name__ == "__main__":
    main()