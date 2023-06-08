# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:02:34 2023

@author: lenovo
"""

from imports import *
from process_data import *
from mixer import mixer_function
from segmentation import generate_masks
from ground_truth import process_ground_truth_data
# from optimizer_function import f
import rasterio
from scipy.optimize import dual_annealing

def print_intermediate_results(x, f, context):
    print('coordinates values are:-', x)
    print('function value of the latest minimum found',f)

    if context == 0:
        print('minimum detected in the annealing process')
    elif context == 1:
        print('detection occurred in the local search process')
    elif context == 2:
        print('detection done in the dual annealing process')

    termination_threshold = 1.5

    if f < termination_threshold:
        print("Termination condition reached: Function value below threashold")
        return True

    # pass
    # print("Current weights: ", xk)

# Read and process raster data
data_filepath = '/home/rajul/Desktop/segment-anything/MergedOutput/20220216_043712_70_2262_3B_AnalyticMS_SR_8b_clip_merged.tif'
data_list = process_data(data_filepath)

# Initialize weights
n = len(data_list)
w_r = np.random.randint(0, 51, n).astype("float64") / np.sum(np.random.randint(0, 51, n).astype("float64"))
w_g = np.random.randint(0, 51, n).astype("float64") / np.sum(np.random.randint(0, 51, n).astype("float64"))
w_b = np.random.randint(0, 51, n).astype("float64") / np.sum(np.random.randint(0, 51, n).astype("float64"))

initial_weights = np.concatenate([w_r, w_g, w_b])

# Define the bounds for each variable
bounds = tuple([(0.0, 1.0)] * len(initial_weights))

# Set the initial temperature and other solver options
initial_temp = 5230
maxiter = 300
solver_options = {'initial_temp': initial_temp, 'maxiter': maxiter, 'seed': 42}

# Set the SAM checkpoint and ground truth file path
sam_checkpoint = "/home/rajul/Desktop/segment-anything/segment_anything/notebooks/sam_vit_h_4b8939.pth"
ground_truth_filepath = '/home/rajul/Desktop/segment-anything/MaskOutput/20220216_043712_70_2262_3B_AnalyticMS_SR_8b_clip_merged_masks.tif'

# initial_weights=[0.20812183, 0.03045685, 0.04568528, 0.2284264,  0.23350254, 0.23857868,0.24873096, 0.03045685, 0.17766497, 0.07482993, 0.05442177, 0.15646259, 0.1122449,  0.03741497, 0.12585034, 0.08163265, 0.01020408, 0.11904762, 0.13537118, 0.17030568, 0.20087336, 0.00436681, 0.21834061, 0.20960699,0.16593886, 0.16157205, 0.12227074]

# Perform optimization using dual_annealing
res = dual_annealing(f, bounds, x0=initial_weights, args=(data_list, sam_checkpoint, ground_truth_filepath), callback=print_intermediate_results, **solver_options)

# Print the results
print(res)
