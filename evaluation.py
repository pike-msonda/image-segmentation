import utils.save_to_folder as stf
from os import listdir
import cv2
from os.path import isfile, join, splitext
import utils.evaluate_boundary as eb
SEG_PATH = "./segs/"
GROUND_TRUTH ="./ground_truth/"

def getTruth():
    # file_list =[f for f in listdir(GROUND_TRUTH) if isfile(join(GROUND_TRUTH, f)) and f.endswith(".mat")]
    file_list = listdir(GROUND_TRUTH)
    seg_list = listdir(SEG_PATH)
    truth = []
    for f, s in zip(file_list, seg_list):
        bound_path = GROUND_TRUTH + f
        seg_path = SEG_PATH + s
        print(bound_path)
        print(seg_path)
        f_truth = stf.read_truth(bound_path)
        img = cv2.imread(seg_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = img.shape
        boundary_predict = img.reshape(size[0], size[1], 1)
        precision, recall = eb.eval_bound(boundary_predict, f_truth, 4)
        print("Precision is: " + str(precision))
        print("Recall is:" + str(recall))

def main():
    getTruth()

if __name__ == "__main__":
    main()