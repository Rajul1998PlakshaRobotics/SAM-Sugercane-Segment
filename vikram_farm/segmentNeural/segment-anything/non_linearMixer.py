# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:12:39 2023

@author: lenovo
"""
import numpy as np
import cv2
from entrophy import multi_band_entropy, calc_color_entropy
import tensorflow as tf

optimization_count_ = 0
mixer_image_count = 0

def nlmixer_function(x, data_list, style_model=tf.keras.applications.DenseNet121(weights='imagenet', include_top=False)):

    global optimization_count_
    optimization_count_ += 1
    print('Value of iteration is:', optimization_count_)
    print('-------------------------INSIDE MIXER------------------------------')

    n = len(data_list)
    red_array = np.zeros_like(data_list[0])
    green_array = np.zeros_like(data_list[0])
    blue_array = np.zeros_like(data_list[0])

    for i in range(n):
        red_array += data_list[i] * x[i]
        green_array += data_list[i] * x[i + n]
        blue_array += data_list[i] * x[i + 2 * n]

    red_array = np.clip(red_array, 0, 255)
    green_array = np.clip(green_array, 0, 255)
    blue_array = np.clip(blue_array, 0, 255)

    mixed_image = (np.dstack((red_array, green_array, blue_array)) * 255.999).astype(np.uint8)
    mixed_image = style_transfer(mixed_image, style_model)

    entropy = multi_band_entropy(mixed_image)

    with open("multi_band_entropy_list.txt", "a") as file:
        file.write("==============================================================\n")
        file.write("Entropy of mixer iteration: {}\n".format(optimization_count_))
        file.write("Entropy: {}\n".format(entropy))
        file.write("==============================================================\n")

    mixer_filename = "mixer{}.png".format(optimization_count_)
    cv2.imwrite(mixer_filename, cv2.cvtColor(mixed_image, cv2.COLOR_BGR2RGB))

    with open("weight.txt", "a") as file:
        file.write("==============================================================\n")
        file.write("Weights for optimization iteration: {}\n".format(optimization_count_))
        file.write("Red Weights: {}\n".format(x))
        file.write("Green Weights: {}\n".format(x[n:2*n]))
        file.write("Blue Weights: {}\n".format(x[2*n:]))
        file.write("==============================================================\n")
    return mixed_image

def style_transfer(image, style_model):
    preprocessed_image = tf.keras.applications.densenet.preprocess_input(image)
    stylized_image = style_model.predict(preprocessed_image)
    stylized_image = stylized_image[0]
    return stylized_image
