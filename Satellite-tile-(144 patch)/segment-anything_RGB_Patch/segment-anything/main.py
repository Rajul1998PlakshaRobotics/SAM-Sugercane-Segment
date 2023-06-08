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
from optimizer_function import f
import rasterio


def print_intermediate_results(xk):
    pass
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
#initial_weights = res.x

print('INITIAL WEIGHTS ARE :-', initial_weights)

# Set bounds for the optimizer
bounds = tuple([(0.0, 1.0)] * len(initial_weights))

# Set the SAM checkpoint and ground truth file path
sam_checkpoint = "/home/rajul/Desktop/segment-anything/segment_anything/notebooks/sam_vit_h_4b8939.pth"
ground_truth_filepath = '/home/rajul/Desktop/segment-anything/MaskOutput/20220216_043712_70_2262_3B_AnalyticMS_SR_8b_clip_merged_masks.tif'



# Perform optimization
res = minimize(f, initial_weights, args=(data_list, sam_checkpoint, ground_truth_filepath), method='SLSQP', bounds=bounds,callback=print_intermediate_results,tol=1e-9)

# print('Total value of optimization count is (Number of iterations SLSQP takes to converge Final error ):-', optimization_count)

# Print the results
print(res)
