# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:12:39 2023

@author: lenovo
"""
import numpy as np
import cv2
from entrophy import multi_band_entropy, calc_color_entropy

optimization_count_ = 0
mixer_image_count = 0

def mixer_function(x, data_list):

    global optimization_count_
    optimization_count_ = optimization_count_ + 1
    print('Value of iteration is:-', optimization_count_)
    print('-------------------------INSIDE MIXER------------------------------')

    n = len(data_list)
    red_array = np.zeros_like(data_list[0])
    green_array = np.zeros_like(data_list[0])
    blue_array = np.zeros_like(data_list[0])

    # for i in range(n):
    #     for j in range(n):
    #         red_array += data_list[i] * x[j]
    #         green_array += data_list[i] * x[j + n]
    #         blue_array += data_list[i] * x[j + 2 * n]

    w_r_mix = np.zeros(n)
    w_g_mix = np.zeros(n)
    w_b_mix = np.zeros(n)

    for i in range(n):
        w_r_mix[i] = x[i]
        w_g_mix[i] = x[i + n]
        w_b_mix[i] = x[i + 2 * n]

    # w_r_mix = w_r_mix/np.sum(w_r_mix.astype("float64"))
    # w_g_mix = w_g_mix/np.sum(w_g_mix.astype("float64"))
    # w_b_mix = w_b_mix/np.sum(w_b_mix.astype("float64"))

    w_r_mix = w_r_mix/np.sqrt(np.sum(w_r_mix**2)).astype("float64")
    w_g_mix = w_g_mix/np.sqrt(np.sum(w_g_mix**2)).astype("float64")
    w_b_mix = w_b_mix/np.sqrt(np.sum(w_b_mix**2)).astype("float64")

    print('The value of NORMALIZED MIXER [R] weights in this iteration is:-', w_r_mix)
    print('The value of NORMALIZED MIXER [G] weights in this iteration is:-', w_g_mix)
    print('The value of NORMALIZED MIXER [B] weights in this iteration is:-', w_b_mix)

    for i in range(n):
        red_array += data_list[i] * w_r_mix[i]
        green_array += data_list[i] * w_g_mix[i]
        blue_array += data_list[i] * w_b_mix[i]

    # print('The weights are:- ',x)
    global mixer_image_count
    mixer_filename = "mixer"
    mixer_image_count = mixer_image_count + 1
    mixer_filename += str(mixer_image_count)
    mixer_filename += ".png"
    
    mixed_image = (np.dstack((red_array, green_array, blue_array)) * 255.999).astype(np.uint8)
    entropy = multi_band_entropy(mixed_image) 
    # entropy = calc_color_entropy(mixed_image)

    with open("multi_band_entropy_list.txt", "a") as file:
        file.write("==============================================================")
        file.write("entrophy of mixer iteration:  ")
        file.write(str(optimization_count_))
        file.write(' ')
        file.write("is:  ")
        file.write(str(entropy))
        file.write('\n')
        file.write("==============================================================")

    cv2.imwrite(mixer_filename,cv2.cvtColor(mixed_image, cv2.COLOR_BGR2RGB))

    with open("weight.txt", "a") as file:
        file.write("==============================================================")
        file.write("Weights for optimization iteration:  ")
        file.write(str(optimization_count_))
        file.write('\n')
        file.write("R  ")
        file.write(str(w_r_mix))
        file.write('\n')
        file.write("G  ")
        file.write(str(w_g_mix))
        file.write('\n')
        file.write("B  ")
        file.write(str(w_b_mix))
        file.write('\n')  
        file.write("==============================================================")

    cv2.imwrite(mixer_filename,cv2.cvtColor(mixed_image, cv2.COLOR_BGR2RGB))


    return mixed_image