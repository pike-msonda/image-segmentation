#!/usr/bin/python3
from __future__ import division  
import numpy as np
import copy
def helper(b1,b2,h,w,thres):
    cnt = 0
    for i in range(h):
        for j in range(w):
            if b1[i][j]:
                lower_x = max(0,i-thres)
                upper_x = min(h-1,i + thres)
                lower_y = max(0,j-thres)
                upper_y = min(w-1,j + thres)
                matrix_rows = b2[lower_x : upper_x + 1, :]
                matrix = matrix_rows[:, lower_y : upper_y+1]
                if matrix.sum() > 0:
                    cnt = cnt + 1
    total = b1.sum()
    return cnt/ total

    

def eval_bound(mask1, mask2 ,thres):
    '''Evaluate precision for boundary detection'''
    s1 = mask1.shape
    s2 = mask2.shape

    if s1 != s2:
        print ( 'shape not match')
        return -1, -1
    if len(s1) == 3:
        b1 = mask1.reshape( s1[0], s1[1]) == 0
        b2 = mask2.reshape( s2[0], s2[1]) == 0
    else :
        b1 = mask1 == 0
        b2 = mask2 == 0
    
    h = s1[0]
    w = s1[1]
    precision = helper(b1,b2,h,w,thres)  
    recall = helper(b2,b1,h,w,thres)
    return precision, recall
    


