import utils.save_to_folder as stf
from os import listdir
import cv2
from os.path import isfile, join, splitext
import utils.evaluate_boundary as eb
SEG_PATH = "./segs/"
BINARY_PATH = "./binary/"
GROUND_TRUTH ="./ground_truth/"

def getTruth():
    truth_list  =[f for f in listdir(GROUND_TRUTH) if isfile(join(GROUND_TRUTH, f)) and f.endswith(".mat")]
    seg_list = listdir(SEG_PATH)
    
    for sl in seg_list:
        path = SEG_PATH + sl + '/'
        print("======================================")
        print("CALCULATING Normalized Probabilistic Rand (NPR) index FOR "+ sl.upper())
        print("======================================")
        for f, s in zip(truth_list, [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".jpg")]):
                ground_path = GROUND_TRUTH + f
                boundary_path = path + s
                print(ground_path)
                print(boundary_path)
                f_truth = stf.read_truth(ground_path)
                img = cv2.imread(boundary_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                size = img.shape
                boundary_predict = img.reshape(size[0], size[1], 1)
                # import pdb; pdb.set_trace()
                precision, recall = eb.eval_bound(boundary_predict, f_truth, 2)
                print("Precision is: " + str(precision))
                print("Recall is:" + str(recall))

def main():
    getTruth()


if __name__ == "__main__":
    main()