# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:13:58 2023

@author: lenovo
"""
from mixer import mixer_function
from segmentation import generate_masks
from ground_truth import process_ground_truth_data
import numpy as np

optimization_count = 0
append_patch = []

def f(x, data_list, sam_checkpoint, ground_truth_filepath,patch_count):
    append_patch.append(patch_count)
    global optimization_count
    if patch_count == 55 and optimization_count ==0:
        print('Patch count in optimization is ', patch_count)
        
        optimization_count = optimization_count + 1
        print('Value of Optimization count in this iteration is:-', optimization_count)
        print('-------------------------INSIDE OPTIMIZER--------------------------------------')
        
        rgb_mixer_image = mixer_function(x, data_list,patch_count)
        masks = generate_masks(rgb_mixer_image,patch_count, sam_checkpoint)
        mask_list = process_ground_truth_data(ground_truth_filepath,patch_count)

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
        final_error = (len(mask_list)/ np.sum(max_score_vector))
        # final_error = (46 / np.sum(max_score_vector)+0.25*np.count_nonzero(score_matrix)/46)/2
    
        error_filename = "error"
        error_filename += str(patch_count)
        error_filename += ".txt"

        with open(error_filename, "a") as file:
            file.write("==============================================================")
            file.write("Final Error Value for optimization iteration: ")
            file.write(str(optimization_count))
            file.write('\n')
            file.write(str(final_error))
            file.write('\n')
            file.write("==============================================================")

        print('Final Error', final_error)
        return final_error
    
    else:
        if append_patch[-1] != append_patch[-2]:
            optimization_count = 0
            print('Patch count in optimization is ', patch_count)
            
            optimization_count = optimization_count + 1
            print('Value of Optimization count in this iteration is:-', optimization_count)
            print('-------------------------INSIDE OPTIMIZER--------------------------------------')
            
            rgb_mixer_image = mixer_function(x, data_list,patch_count)
            masks = generate_masks(rgb_mixer_image,patch_count, sam_checkpoint)
            mask_list = process_ground_truth_data(ground_truth_filepath,patch_count)

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
            final_error = (len(mask_list)/ np.sum(max_score_vector))
            # final_error = (46 / np.sum(max_score_vector)+0.25*np.count_nonzero(score_matrix)/46)/2
        
            error_filename = "error"
            error_filename += str(patch_count)
            error_filename += ".txt"

            with open(error_filename, "a") as file:
                file.write("==============================================================")
                file.write("Final Error Value for optimization iteration: ")
                file.write(str(optimization_count))
                file.write('\n')
                file.write(str(final_error))
                file.write('\n')
                file.write("==============================================================")

            print('Final Error', final_error)

            return final_error
        else:
            print('Patch count in optimization is ', patch_count)

            optimization_count = optimization_count + 1
            print('Value of Optimization count in this iteration is:-', optimization_count)
            print('-------------------------INSIDE OPTIMIZER--------------------------------------')
            
            rgb_mixer_image = mixer_function(x, data_list,patch_count)
            masks = generate_masks(rgb_mixer_image,patch_count, sam_checkpoint)
            mask_list = process_ground_truth_data(ground_truth_filepath,patch_count)

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
            final_error = (len(mask_list)/ np.sum(max_score_vector))
            # final_error = (46 / np.sum(max_score_vector)+0.25*np.count_nonzero(score_matrix)/46)/2
        
            error_filename = "error"
            error_filename += str(patch_count)
            error_filename += ".txt"

            with open(error_filename, "a") as file:
                file.write("==============================================================")
                file.write("Final Error Value for optimization iteration: ")
                file.write(str(optimization_count))
                file.write('\n')
                file.write(str(final_error))
                file.write('\n')
                file.write("==============================================================")

            print('Final Error', final_error)

            return final_error
            
