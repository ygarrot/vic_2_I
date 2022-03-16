from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from gc import set_threshold
import sys, traceback
import cv2 as cv
import os
import re
import numpy as np
import argparse
import string
from plantcv import plantcv as pcv
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

lame = [
"Apple___Apple_scab/image (136).JPG",
"Apple___Apple_scab/image (619).JPG",
"Apple___Apple_scab/image (799).JPG",
"Pepper,_bell___Bacterial_spot/image (17).JPG",
"Pepper,_bell___Bacterial_spot/image (887).JPG",
"Corn___healthy/image (349).JPG",
"Corn___healthy/image (694).JPG",
"Corn___Northern_Leaf_Blight/image (288).JPG",
"Corn___Northern_Leaf_Blight/image (12).JPG",
"Corn___Northern_Leaf_Blight/image (645).JPG",
"Corn___Cercospora_leaf_spot Gray_leaf_spot/image (112).JPG",
"Corn___Cercospora_leaf_spot Gray_leaf_spot/image (266).JPG",
"Corn___healthy/image (123).JPG",
"Peach___Bacterial_spot/image (703).JPG",
"Peach___Bacterial_spot/image (2007).JPG",
"Peach___healthy/image (34).JPG",
"Potato___Early_blight/image (700).JPG",
"Potato___Late_blight/image (45).JPG",
"Strawberry___Leaf_scorch/image (685).JPG",
"Strawberry___Leaf_scorch/image (912).JPG",
"Strawberry___Leaf_scorch/image (282).JPG",
]
# plt.rcParams["figure.figsize"] = (20,9)
# pcv.params.debug = 'plot' 

# img=images[47]

import random
def save_img(img, img_path):
    p = (img_path).replace("resources", "generate")
    print(p)
    plt.imshow(img)
    plt.savefig(p)

def plot_images(ar):
    l = (len(ar) / 2)  + 1
    fig = plt.figure(figsize=(20, 9))
    for i, k in enumerate(ar):
        fig.add_subplot(l, l, i+1)
        plt.axis('off')
        plt.title(k)
        plt.imshow(ar[k])
    plt.show()

root_path = "../resources/Plant_leave_diseases_dataset_without_augmentation/"

# x=widgets.IntSlider(min=-10, max=30, step=1, value=10)
# @interact(
#     i_b_thresh = 147,
#     i_b_cnt = 112,
#     masked_a_thresh = 125,
#     masked_a1_thresh = 136,
#     masked_b_thresh = 132,
#     index = widgets.IntSlider(min=1, max=150, step=1, value=1),
# )
def tutorial(
    img_path,
    i_b_thresh = 147,
    i_b_cnt = 112,
    masked_a_thresh = 127,
    masked_a1_thresh = 136,
    masked_b_thresh = 133,
    # index = 133,
):
    # r_i = random.randrange(0, len(lame))
    # img_path = without_path + lame[index]

    ## example
    img, path, filename = pcv.readimage(img_path, mode='rgb')
    gray        = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    s_thresh = pcv.threshold.binary(gray_img=gray, threshold=36, max_value=255, object_type='light')

    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    s_cnt = pcv.median_blur(gray_img=s_thresh, ksize=5)

    fill     = pcv.fill_holes(bin_img=s_mblur)
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')

    # Threshold the blue image

    b_thresh = pcv.threshold.binary(gray_img=b, threshold=i_b_thresh, max_value=255, 
                                    object_type='light')
    b_cnt = pcv.threshold.binary(gray_img=b, threshold=i_b_cnt, max_value=255, 
                                 object_type='light')

    b_fill = pcv.fill(b_thresh, 10)

    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_cnt)
    masked = pcv.apply_mask(img, mask=bs, mask_color='white')

    ##########################################################################
    # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
    masked_a = pcv.rgb2gray_lab(masked, channel='a')
    masked_b = pcv.rgb2gray_lab(masked, channel='b')
    
    # Threshold the green-magenta and blue images
    maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=masked_a_thresh, 
                                      max_value=255, object_type='dark')
    maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=masked_a1_thresh, 
                                           max_value=255, object_type='light')
    maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=masked_b_thresh, 
                                          max_value=255, object_type='light')
    ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

    ab_fill = pcv.fill(bin_img=ab, size=200)

    # Apply mask (for VIS images, mask_color=white)
    masked2 = pcv.apply_mask(masked, mask=ab_fill, mask_color='white')
    save_img(masked2, img_path)
    # steps = {
    #     "maskeda_thresh ": maskeda_thresh,
    #     "maskeda_thresh1": maskeda_thresh1,
    #     "maskedb_thresh ": maskedb_thresh,
    #     "masked2 ": masked2,
    #     "img ": img,
    # }
    # plot_images(steps)

# images = glob.glob(without_path + 'Corn*/*')
images = glob.glob(root_path + 'Corn*/*')
for img_path in images:
    # img_path = without_path + img_path
    tutorial(img_path)
    # img_path = images[index]
    #img, path, filename = pcv.readimage(img_path, mode='rgb')

    #image_analysis = pcv.visualize.colorspaces(rgb_img=img)
    #color_analysis = pcv.analyze_color(rgb_img=img, mask=None, colorspaces='all', label="default")

    #gray        = pcv.rgb2gray_hsv(rgb_img=img, channel='s')



    ######BLUR##############
    #blur        = pcv.gaussian_blur(img=gray, ksize=(11, 11), sigma_x=0, sigma_y=None)
    #s_thresh = pcv.threshold.binary(gray_img=gray, threshold=85, max_value=255, object_type='light')
    #s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    #s_cnt = pcv.median_blur(gray_img=s_thresh, ksize=5)
    ######END BLUR##############

    #otsu        = pcv.threshold.otsu(gray_img=blur, max_value=255, object_type='light')
    #kernel      = np.ones((20,20),np.uint8)
    #morphology  = cv.morphologyEx(otsu, cv.MORPH_CLOSE, kernel)

    #closing     = cv.dilate(morphology, kernel, iterations = 1)
    #closing     = pcv.fill_holes(bin_img=closing)

    #and_pcv     = pcv.logical_and(gray, closing)
    #and_cv      = cv.bitwise_and(img, img, mask = closing)

    #steps = {
    #    "image_analysis":         image_analysis,
    #    # "color_analysis":         color_analysis,
    #}
    #plot_images(steps)
    #steps = {
    #    "orig":         img,
    #    "gray":         gray,
    #    "blur":         blur,
    #    "otsu":         otsu,
    #    "morphology":   morphology,
    #    "closing":      closing,
    #    "and_cv":       and_cv,
    #    "and_pcv":      and_pcv,
    #}
    #plot_images(steps)


