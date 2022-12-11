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

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder("C:\\data\\test")

# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

def makeSubplot(image, ax, key, colormap="gray", title=""):
    ax[key].imshow(image, cmap=colormap)
    ax[key].set_title(title)
    ax[key].axis('off')

# loop over the input image paths
for image in images:
    height, width, _ = image.shape

    myDpi = 96
    fig, ax = plt.subplot_mosaic([
        ['Original', 'Resized', "Gray"]
        ], figsize=(1600/myDpi, 1000/myDpi), dpi=myDpi)
    makeSubplot(image, ax, "Original", title="Original")

    # Resize to uniform width
    image = imutils.resize(image, width=600)
    makeSubplot(image, ax, "Resized", title="Resized")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    makeSubplot(gray, ax, "Gray", title="Gray")

    myDpi = 96
    fig, ax = plt.subplot_mosaic([
        ['Smoothed', "Blackhat", "Gradient"]
        ], figsize=(1600/myDpi, 1000/myDpi), dpi=myDpi)

	# smooth the image using a 3x3 Gaussian to reduce high frequeny noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    makeSubplot(gray, ax, "Smoothed", title="Smoothed")

    # Blackhat - enhances dark objects of interest in a bright background.
    # The black-hat transform is defined as the difference between the closing and the input image.
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    makeSubplot(blackhat, ax, "Blackhat", title="Blackhat")

    # compute the Scharr gradient of the blackhat image in the x direction and scale the
	# result into the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    makeSubplot(gradX, ax, "Gradient", title="Gradient")

    # # compute the Scharr gradient of the blackhat image  in the y direction and scale the
	# # result into the range [0, 255]
    # gradY = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # gradY = np.absolute(gradY)
    # (minVal, maxVal) = (np.min(gradY), np.max(gradY))
    # gradY = (255 * ((gradY - minVal) / (maxVal - minVal))).astype("uint8")
    # makeSubplot(gradY, ax, "Gradient", title="Gradient")

plt.show()
