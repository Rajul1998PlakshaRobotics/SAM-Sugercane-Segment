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
from scipy.optimize import dual_annealing
import tifffile as tif
import itertools
import concurrent.futures
import multiprocessing
import time
import shutil

parent_folder_path = '/home/rajul/Desktop/segment-anything/CroppedOutput/2022-06-09_227c_BGRN'

mask_folder = 'mask12'
raster_folder = 'raster12'
score_minimum = []
matching_files = []

mask_folder = os.path.join(parent_folder_path, mask_folder)
raster_folder = os.path.join(parent_folder_path, raster_folder)

if os.path.exists(mask_folder) and os.path.isdir(mask_folder) and os.path.exists(raster_folder) and os.path.isdir(raster_folder):
    for number in range(55,145):
        number_str = str(number).zfill(2)

        mask = f"piece_{number_str}_mask.tif"
        input = f"piece_{number_str}.tif"

        mask = os.path.join(mask_folder, mask)
        input = os.path.join(raster_folder, input)

        if os.path.isfile(mask) and os.path.isfile(input):
            matching_files.append((mask, input))
else:
    print("Mask and raster files are not equal")

#------------------------------------------------------------------------------

patch_count = 54

def patch_appending(score_inpatch_,patch_count_):
        score_minimum.append(score_inpatch_)

        print('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC')

        folder_path = '/home/rajul/Desktop/segment-anything'
        new_folder_name = "patchResults_"
        new_folder_name += str(patch_count_)

        print('ddddddddddddddddddddddddddddddddddddddddddddddddd')
        # Create a new folder inside the main folder
        new_folder_path = os.path.join(folder_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # Find PNG images in the main folder
        png_image_paths = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith('.png'):
                png_image_paths.append(file_path)

        print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
        # Move PNG images to the new folder
        for png_image_path in png_image_paths:
            shutil.move(png_image_path, os.path.join(new_folder_path, os.path.basename(png_image_path)))
        print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')


def doingSAM(rasterMasks_path):
        global patch_count
        patch_count += 1
        print('Value of patch count in main is:', patch_count)
        
        data_list = process_data(rasterMasks_path[1])
        data_list = np.array(data_list,dtype=np.float64)

        # print('len of data list', len(data_list))

        # n = len(data_list)
        # print('n is:-',n)

        initial_weights = [0,0,1,0,0,1,0,0,1,0,0,0]
        pattern_length = len(initial_weights)

        bounds = tuple([(-1.0, 1.0)] * len(initial_weights))

        # Set the SAM checkpoint and ground truth file path
        sam_checkpoint = "/home/rajul/Desktop/segment-anything/segment_anything/notebooks/sam_vit_h_4b8939.pth"
        ground_truth_filepath = rasterMasks_path[0]

        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

        res = dual_annealing(f, bounds, x0=initial_weights, args=(data_list, sam_checkpoint, ground_truth_filepath,patch_count),seed= 42,initial_temp = 5230,maxiter = 400, maxfun=400)
        print('error is' ,res.fun) 
        score_inpatch = res.fun
        print('score in patch',score_inpatch)
        print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
              
        patch_appending(score_inpatch,patch_count)
        # Print the results


for tuple_data in matching_files:
    doingSAM(tuple_data)
# pool = multiprocessing.Pool()
# pool = multiprocessing.Pool(processes=4)
# inputs = matching_files
# outputs = pool.map(doingSAM, inputs)
# print("Input: {}".format(inputs))
# print("Output: {}".format(outputs))
print("Count is", patch_count)

#------------------------------------------------------------------------------

print('score list is',  score_minimum)
score_minimum_without_inf = [x for x in score_minimum if x != float('inf')]
max_value = max(score_minimum_without_inf)
max_index = score_minimum.index(max_value)

min_value = min(score_minimum_without_inf)
min_index = score_minimum.index(min_value)

average_score = np.mean(score_minimum_without_inf)

print('Avaerage score is',  average_score)

print('Minimum score is:-',min_value)
print('minimum score is for this peice', (score_minimum.index(min_value))+1)

print('Maximum score is:-',max_value)
print('maximum score is for this peice', (score_minimum.index(max_value))+1)

with open("minumum_score_file.txt", "a") as file:
    file.write("==============================================================")
    file.write("Avaerage score is  ")
    file.write(str(average_score))
    file.write('\n')
    file.write("Minimum score is:  ")
    file.write(str(min_value))
    file.write('\n')
    file.write("minimum score is for this peice  ")
    file.write(str((score_minimum.index(min_value))+1))
    file.write('\n')
    file.write("Maximum score is:  ")
    file.write(str(max_value))
    file.write('\n')  
    file.write("maximum score is for this peice  ")
    file.write(str((score_minimum.index(max_value))+1))
    file.write("==============================================================")





