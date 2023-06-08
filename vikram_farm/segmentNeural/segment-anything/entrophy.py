import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import rasterio
import os
import torch
import torchvision
import glob
from PIL import Image
from scipy.optimize import minimize
from os import listdir
import tifffile as tif
import concurrent.futures
import shutil
import math
import rasterio

def multi_band_entropy(mixer_image_):

    channels = mixer_image_.shape[2]
    print('No. of channels is: ', channels)
    entropy = np.zeros(channels)

    for i in range(channels):
        channel = mixer_image_[:,:,i]
        counts, _ = np.histogram(channel.flatten(), bins=256, range=[0,256])
        probabilities = counts / np.sum(counts)
        entropy[i] = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
        print('The entrophy is: ', entropy[i])
    average = sum(entropy)/len(entropy)
    return average


def calc_color_entropy(img_fused):
    image = img_fused * 256
    num_bins = 4
    histogram = np.zeros((num_bins, num_bins, num_bins))
    rows, cols, _ = image.shape
    temp_histogram = []
    print('Rows', rows)
    print('Column', cols)

    for i in range(rows):
        temp = np.zeros((num_bins, num_bins, num_bins))
        for j in range(cols):
            r = min(int(image[i, j, 0] / (256/num_bins)), num_bins-1)
            g = min(int(image[i, j, 1] / (256/num_bins)), num_bins-1)
            b = min(int(image[i, j, 2] / (256/num_bins)), num_bins-1)
            temp[r, g, b] += 1
        temp_histogram.append(temp)

    for i in range(rows):
        histogram += temp_histogram[i]

    histogram /= np.sum(histogram)
    histogram = histogram.flatten()
    entropy = -np.sum(histogram * np.log2(histogram + np.finfo(float).eps))
    print('The entrophy is: ', entropy)
    return entropy




