# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:13:58 2023

@author: lenovo
"""
from mixer import mixer_function
from segmentation import generate_masks
from ground_truth import process_ground_truth_data
import numpy as np
import cv2

optimization_count = 0

def doing(dataR, dataG, dataB, n, data_list, sam_checkpoint, ground_truth_filepath):

    global optimization_count
    optimization_count = optimization_count + 1
    print('Value of Optimization count in this iteration is:-', optimization_count)
    print('-------------------------INSIDE OPTIMIZER--------------------------------------')
    
    rgb_mixer_image = mixer_function(dataR, dataG, dataB, data_list)
    rgb_mixer_image1 = cv2.imread('/home/rajul/Desktop/segment-anything/input_images/schaarRGB_0bLUR_EdgeDetection.png')
    masks = generate_masks(rgb_mixer_image1, sam_checkpoint)
    mask_list = process_ground_truth_data(ground_truth_filepath)

    score_matrix = np.zeros((len(mask_list), len(masks)))
    score_matrix1 = np.zeros((len(mask_list), len(masks)))
    score_matrix2 =np.zeros((len(mask_list), len(masks)))

    for i in range(len(mask_list)):
        for j in range(len(masks)):
            score_image = np.multiply(mask_list[i], masks[j]["segmentation"])
            sum_G=np.sum(mask_list[i])
            sum_M=np.sum(masks[j]["segmentation"])
            eq1=np.sum(score_image) / sum_G
            eq2=np.sum(score_image) / sum_M
            # score_matrix[i, j] = np.sum(score_image) / np.sum(mask_list[i])
            score_matrix[i, j] = min(eq1,eq2)
            score_matrix1[i, j] = eq1
            score_matrix2[i, j] = eq2

    max_score_vector = [max(row) for row in score_matrix]
    final_error = (46 / np.sum(max_score_vector))
    # final_error = (46 / np.sum(max_score_vector)+0.25*np.count_nonzero(score_matrix)/46)/2

    with open("error.txt", "a") as file:
        file.write("==============================================================")
        file.write("Final Error Value for optimization iteration: ")
        file.write(str(optimization_count))
        file.write('\n')
        file.write(str(final_error))
        file.write('\n')
        file.write("==============================================================")

    print('Final Error', final_error)


    return final_error
