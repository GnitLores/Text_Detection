from turtle import color
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
from TextDetector import *


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder("C:\\data\\test")

# initialize a rectangular and square structuring kernel
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

def makeSubplot(image, ax, key, colormap="gray", title=""):
    ax[key].imshow(image, cmap=colormap)
    ax[key].set_title(title)
    ax[key].axis('off')

# loop over the input image paths
for image in images:
    textDetector = TextDetector(image, do_visualize = True)
    textDetector.detect_text()