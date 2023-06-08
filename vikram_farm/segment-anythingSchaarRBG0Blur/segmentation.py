# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:13:07 2023

@author: lenovo
"""
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np

seg_image_count = 0

def generate_masks(rgb_mixer_image, sam_checkpoint, device="cuda", model_type="default"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(rgb_mixer_image)

    print('No of segments generated by FB algo is:-', len(masks))

    save_anns(masks)
    for mask in masks:
        segmentation = mask["segmentation"]
        segmentation[segmentation > 0] = 1
        segmentation[segmentation < 0] = 0

    return masks

def save_anns(anns, output_filename=""):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # Initialize an empty image
    m = anns[0]['segmentation']
    mask_image = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.float32)

    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))

        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]

        # Add the colored mask with 35% opacity to the image
        mask_image += img * m[:, :, np.newaxis] * 0.35

    global seg_image_count
    output_filename="seg"
    seg_image_count = seg_image_count + 1
    output_filename += str(seg_image_count)
    output_filename += ".png"
    # Clip the final image to the valid range [0, 1] and convert it to the format of the original image
    mask_image = np.clip(mask_image, 0, 1)

    # Save the image
    cv2.imwrite(output_filename, cv2.cvtColor((mask_image * 255.999).astype(np.uint8), cv2.COLOR_RGB2BGR))