import os

import cv2
from matplotlib import pyplot as plt
from TextDetector import TextDetector


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


images = load_images_from_folder("C:\\data\\test")
# images = load_images_from_folder("C:\\data\\multiple")


def makeSubplot(image, ax, key, colormap="gray", title=""):
    ax[key].imshow(image, cmap=colormap)
    ax[key].set_title(title)
    ax[key].axis("off")


# loop over the input image paths
for image in images:
    textDetector = TextDetector(
        image,
        # Comment line to disable option:
        # show_process=True,
        # show_segments=True,
        show_result=True,
    )
    textDetector.detect_text()
plt.show()
