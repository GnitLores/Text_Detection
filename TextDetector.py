import imutils
import cv2
from matplotlib import pyplot as plt

class TextDetector:
    def __init__(self, image, do_visualize = False, is_text_vertical = True):
        self.original_image = image
        self.do_visualize = do_visualize
        self.vert_text = is_text_vertical
        self.height, self.width, _ = image.shape

        self.processed_width = 600
        
    def detect_text(self):
        self.__preprocessing()

    def __make_subplot_figure(self, subplot_keys):
        dpi = 96
        figure_width = 1600
        figure_height = 1000
        fig, ax = plt.subplot_mosaic([subplot_keys], figsize = (figure_width / dpi, figure_height / dpi), dpi = dpi)
        return fig, ax

    def __makeSubplot(self, image, ax, key, colormap = "gray", title = ""):
        ax[key].imshow(image, cmap=colormap)
        ax[key].set_title(title)
        ax[key].axis('off')

    def __preprocessing(self):
        
        if self.do_visualize: fig, ax = self.__make_subplot_figure(['Original', "Blurred", "Blackhat"])
        if self.do_visualize: self.__makeSubplot(self.original_image, ax, "Original", title="Original")
        
        # Resize to uniform width
        self.image = imutils.resize(self.original_image, width = self.processed_width)

        # Convert to grayscale
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # smooth the image using Gaussian to reduce high frequeny noise
        self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
        if self.do_visualize: self.__makeSubplot(self.image, ax, "Blurred", title="Blurred")

        # Blackhat - enhances dark objects of interest in a bright background.
        # The black-hat transform is defined as the difference between the closing and the input image.
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_BLACKHAT, rectKernel)
        if self.do_visualize: self.__makeSubplot(self.image, ax, "Blackhat", title="Blackhat")

        if self.do_visualize: plt.show()