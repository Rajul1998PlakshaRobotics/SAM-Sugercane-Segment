# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:11:44 2023

@author: lenovo
"""
import rasterio
def read_raster_file(filepath):
    with rasterio.open(filepath) as src:
        data = [src.read(i) for i in range(1, src.count + 1)]
    return data


def remove_negative_values(data):
    data[data < 0] = 0
    return data


def process_data(filepath):
    data_list = read_raster_file(filepath)
    data_list[-1] = remove_negative_values(data_list[-1])
    return data_list
