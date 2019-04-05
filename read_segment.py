import os
import numpy as np

def readSeg():
    seg =  open('33039.seg','r')
    width, height = get_width_height(seg)
    seg.seek(0)
    data = get_data(seg)
    import pdb; pdb.set_trace()
    return data, width, height

def get_width_height(binary):
    width, height = [line.strip() for idx, line in enumerate(binary) if idx in [4,5]]
    width = int(width.split(' ')[1])
    height = int(height.split(' ')[1])
    return width, height

def get_data(binary):
    data = [line.strip() for idx, line in enumerate(binary)][11:1440]
    result = []
    for item in data:
        item = convert_string_array(item)
        result.append(item)
    return np.array(result)

def convert_string_array(item):
    result =[]
    for i in item.split():
        result.append(int(i))
    return np.array(result, dtype=np.int)

if __name__ == "__main__":
    readSeg()