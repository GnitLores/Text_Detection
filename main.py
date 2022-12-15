import cv2
import os
from TextDetector import TextDetector
from matplotlib import pyplot as plt


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# images = load_images_from_folder("C:\\data\\test")
images = load_images_from_folder("C:\\data\\multiple")

def makeSubplot(image, ax, key, colormap="gray", title=""):
    ax[key].imshow(image, cmap=colormap)
    ax[key].set_title(title)
    ax[key].axis('off')

# loop over the input image paths
for image in images:
    textDetector = TextDetector(image)
    # textDetector = TextDetector(image, do_visualize = True)
    # textDetector = TextDetector(image, do_profile = True)
    textDetector.detect_text()
plt.show()