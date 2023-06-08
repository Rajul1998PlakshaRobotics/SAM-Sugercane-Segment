# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:13:34 2023

@author: lenovo
"""
from process_data import *
import cv2
import numpy as np

def process_ground_truth_data(filepath):
    mask_list = read_raster_file(filepath)
    sum=mask_list[0]*0

    for mask in mask_list:
        mask[mask > 0] = 1
        mask[mask < 0] = 0
        sum=sum+mask
        
    grayscale_image = (sum * 255).astype(np.uint8)
    cv2.imwrite("ground_truth.png", grayscale_image)



    return mask_list
