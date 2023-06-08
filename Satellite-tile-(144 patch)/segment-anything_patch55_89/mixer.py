# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:12:39 2023

@author: lenovo
"""
import numpy as np
import cv2

mixer_image_count = 0
append_mixpatch = []

def mixer_function(x, data_list,patch_count):
    append_mixpatch.append(patch_count)
    global mixer_image_count
    if patch_count == 55 and mixer_image_count == 0:
        
        mixer_image_count = mixer_image_count + 1
        print('Value of mixer iteration is:-', mixer_image_count)
        print('-------------------------INSIDE MIXER--------------------------------------')

        n = len(data_list)
        red_array = np.zeros_like(data_list[0])
        green_array = np.zeros_like(data_list[0])
        blue_array = np.zeros_like(data_list[0])

        w_r_mix = np.zeros(n)
        w_g_mix = np.zeros(n)
        w_b_mix = np.zeros(n)

        for i in range(n):
            w_r_mix[i] = x[i]
            w_g_mix[i] = x[i + n]
            w_b_mix[i] = x[i + 2 * n]

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

        def convert(img, target_type_min, target_type_max, target_type):
            imin = img.min()
            imax = img.max()
            a = (target_type_max - target_type_min) / (imax - imin)
            b = target_type_max - a * imax
            new_img = (a * img + b).astype(target_type)
            return new_img
        
        mixed_image=(np.dstack((red_array, green_array, blue_array)))   #* 255.999).astype(np.uint8) #
        mixed_image= convert(mixed_image, 0, 255, np.uint8)

        weight_filename = "weight"
        weight_filename += str(patch_count)
        weight_filename += ".txt"

        with open(weight_filename, "a") as file:
            file.write("==============================================================")
            file.write("Weights for optimization iteration:  ")
            file.write(str(mixer_image_count))
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

        
        mixer_filename = "mixer"
        mixer_filename += str(mixer_image_count)
        mixer_filename += ".png"

        cv2.imwrite(mixer_filename,cv2.cvtColor(mixed_image, cv2.COLOR_BGR2RGB))


        return mixed_image
    else:
        if append_mixpatch[-1] != append_mixpatch[-2]:
            mixer_image_count = 0
            
            mixer_image_count = mixer_image_count + 1
            print('Value of mixer iteration is:-', mixer_image_count)
            print('-------------------------INSIDE MIXER--------------------------------------')

            n = len(data_list)
            red_array = np.zeros_like(data_list[0])
            green_array = np.zeros_like(data_list[0])
            blue_array = np.zeros_like(data_list[0])

            w_r_mix = np.zeros(n)
            w_g_mix = np.zeros(n)
            w_b_mix = np.zeros(n)

            for i in range(n):
                w_r_mix[i] = x[i]
                w_g_mix[i] = x[i + n]
                w_b_mix[i] = x[i + 2 * n]

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

            def convert(img, target_type_min, target_type_max, target_type):
                imin = img.min()
                imax = img.max()
                a = (target_type_max - target_type_min) / (imax - imin)
                b = target_type_max - a * imax
                new_img = (a * img + b).astype(target_type)
                return new_img
            
            mixed_image=(np.dstack((red_array, green_array, blue_array)))   #* 255.999).astype(np.uint8) #
            mixed_image= convert(mixed_image, 0, 255, np.uint8)

            weight_filename = "weight"
            weight_filename += str(patch_count)
            weight_filename += ".txt"

            with open(weight_filename, "a") as file:
                file.write("==============================================================")
                file.write("Weights for optimization iteration:  ")
                file.write(str(mixer_image_count))
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

            
            mixer_filename = "mixer"
            mixer_filename += str(mixer_image_count)
            mixer_filename += ".png"

            cv2.imwrite(mixer_filename,cv2.cvtColor(mixed_image, cv2.COLOR_BGR2RGB))


            return mixed_image
        else:
            
            mixer_image_count = mixer_image_count + 1
            print('Value of mixer iteration is:-', mixer_image_count)
            print('-------------------------INSIDE MIXER--------------------------------------')

            n = len(data_list)
            red_array = np.zeros_like(data_list[0])
            green_array = np.zeros_like(data_list[0])
            blue_array = np.zeros_like(data_list[0])

            w_r_mix = np.zeros(n)
            w_g_mix = np.zeros(n)
            w_b_mix = np.zeros(n)

            for i in range(n):
                w_r_mix[i] = x[i]
                w_g_mix[i] = x[i + n]
                w_b_mix[i] = x[i + 2 * n]

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

            def convert(img, target_type_min, target_type_max, target_type):
                imin = img.min()
                imax = img.max()
                a = (target_type_max - target_type_min) / (imax - imin)
                b = target_type_max - a * imax
                new_img = (a * img + b).astype(target_type)
                return new_img
            
            mixed_image=(np.dstack((red_array, green_array, blue_array)))   #* 255.999).astype(np.uint8) #
            mixed_image= convert(mixed_image, 0, 255, np.uint8)

            weight_filename = "weight"
            weight_filename += str(patch_count)
            weight_filename += ".txt"

            with open(weight_filename, "a") as file:
                file.write("==============================================================")
                file.write("Weights for optimization iteration:  ")
                file.write(str(mixer_image_count))
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

            
            mixer_filename = "mixer"
            mixer_filename += str(mixer_image_count)
            mixer_filename += ".png"

            cv2.imwrite(mixer_filename,cv2.cvtColor(mixed_image, cv2.COLOR_BGR2RGB))


            return mixed_image
            