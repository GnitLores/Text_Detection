import cv2
import os
from TextDetector import TextDetector
from matplotlib import pyplot as plt


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


# Comment line to disable option:
# show_process = True
# show_segments = True
show_result = True

if "show_process" not in locals():
    show_process = False
if "show_segments" not in locals():
    show_segments = False
if "show_result" not in locals():
    show_result = False

# loop over the input image paths
for image in images:
    textDetector = TextDetector(
        image,
        show_process=show_process,
        show_segments=show_segments,
        show_result=show_result,
    )
    textDetector.detect_text()
plt.show()
