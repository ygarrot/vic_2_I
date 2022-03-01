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
# pcv.params.debug = 'plot' 

images = glob.glob('../resources/Plant_leave_diseases_dataset_without_augmentation/Cherry*/*')
# img=images[47]
# print(img)
for img_path in images:
    img, path, filename = pcv.readimage(img_path)

    gray_img = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    b_gi = pcv.gaussian_blur(img=gray_img, ksize=(11, 11), sigma_x=0, sigma_y=None)

    cropped_mask = pcv.threshold.otsu(gray_img=b_gi, max_value=255, object_type='light')

    kernel = np.ones((20,20),np.uint8)
    closing = cv.morphologyEx(cropped_mask, cv.MORPH_CLOSE, kernel)

    and_image = pcv.logical_and(gray_img, closing)
    img1_bg = cv.bitwise_and(img,img,mask = closing)
    p = (img_path).replace("resources", "generate")
    print(p)
    plt.imshow(img1_bg)
    plt.savefig(p)
