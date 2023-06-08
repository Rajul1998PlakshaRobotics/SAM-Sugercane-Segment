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
from optimizer_function import doing
import rasterio
from scipy.optimize import dual_annealing
import tifffile as tif
import itertools

parent_folder_path = '/home/rajul/Desktop/segment-anything/CroppedOutput/2022-01-28_241c_BGRN'

mask_folder = 'mask1'
raster_folder = 'raster1'
score = []
# Construct the full paths for folder1 and folder2
mask_folder = os.path.join(parent_folder_path, mask_folder)
raster_folder = os.path.join(parent_folder_path, raster_folder)

# Check if folder1_path and folder2_path are valid directories
if os.path.exists(mask_folder) and os.path.isdir(mask_folder) and os.path.exists(raster_folder) and os.path.isdir(raster_folder):
    # Iterate over the range of numbers from 1 to 144
    for number in range(1, 145):
        # Format the number with leading zero if necessary
        number_str = str(number).zfill(2)

        # Construct the file names with the common numbering pattern
        mask = f"piece_{number_str}_mask.tif"
        input = f"piece_{number_str}.tif"

        mask = os.path.join(mask_folder, mask)
        input = os.path.join(raster_folder, input)

        # Check if both file paths exist and are valid files
        if os.path.isfile(mask) and os.path.isfile(input):
            print("Matching files found:")
            print("Folder 1:", mask)
            print("Folder 2:", input)
            print()

            data_list = process_data(input)
            data_list = np.array(data_list,dtype=np.float64)

            print('len of data list', len(data_list))

            # Initialize weights
            n = len(data_list)
            print('n is:-',n)

            initial_weights = [0,0,1,0,0,1,0,0,1,0,0,0]
            pattern_length = len(initial_weights)

            # Set the SAM checkpoint and ground truth file path
            sam_checkpoint = "/home/rajul/Desktop/segment-anything/segment_anything/notebooks/sam_vit_h_4b8939.pth"
            ground_truth_filepath = mask

            res = doing(initial_weights,data_list, sam_checkpoint, ground_truth_filepath)

            score.append(res)
            # Print the results
            print(res)

# Handle cases where folder paths are invalid or files don't exist
else:
    print("Invalid folder paths or files not found.")

print('score list is',  score)
print('Avaerage score is',  sum(score)/len(score))
min_value = min(score)
max_value = max(score)
print('Minimum score is:-',min_value)
print('minimum score is for this peice', score.index(min_value))

print('Maximum score is:-',max_value)
print('maximum score is for this peice', score.index(max_value))




