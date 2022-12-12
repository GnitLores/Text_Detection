import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt

class TextDetector:
    def __init__(self, image, do_visualize = False, is_text_vertical = True):
        self.original_image = image
        self.do_visualize = do_visualize
        self.vert_text = is_text_vertical
        self.height, self.width, _ = image.shape

        self.processed_width = 600
        
    def detect_text(self):
        self.__preprocess()
        self.__filter_chars()
        self.__process_secondary()
        self.__filter_sentences()
        self.__filter_text_blocks()

        if self.do_visualize: plt.show()

    def __make_subplot_figure(self, subplot_keys, title = ""):
        dpi = 96
        figure_width = 1600
        figure_height = 1000
        fig, ax = plt.subplot_mosaic([subplot_keys], figsize = (figure_width / dpi, figure_height / dpi), dpi = dpi)
        fig.suptitle(title)
        return fig, ax

    def __make_subplot(self, image, ax, key, colormap = "gray", title = ""):
        ax[key].imshow(image, cmap = colormap)
        ax[key].set_title(title)
        ax[key].axis('off')

    def __preprocess(self):
        
        if self.do_visualize:
            key1, key2, key3 = "1", '2', "3"
            fig, ax = self.__make_subplot_figure([key1, key2, key3], title = "1: Preprocessing")
        if self.do_visualize: self.__make_subplot(self.original_image, ax, key1, title = "Original Image")
        
        # Resize to uniform width
        self.image = imutils.resize(self.original_image, width = self.processed_width)

        # Convert to grayscale
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # smooth the image using Gaussian to reduce high frequeny noise
        self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
        if self.do_visualize: self.__make_subplot(self.image, ax, key2, title = "Rescaled, Greyscale, Blurred image")

        # Blackhat - enhances dark objects of interest in a bright background.
        # The black-hat transform is defined as the difference between the closing and the input image.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_BLACKHAT, kernel)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Blackhat Transform")

    def __filter_chars(self):
        
        if self.do_visualize:
            key1, key2, key3 = "1", '2', "3"
            fig, ax = self.__make_subplot_figure([key1, key2, key3], title = "2: Filter Characters")

        # Get black/white image with otsu threshold
        self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if self.do_visualize: self.__make_subplot(self.image, ax, key1, title = "Thresholded image")

        # Create mask of components that are very unlike characters
        analysis = cv2.connectedComponentsWithStats(self.image, 4, cv2.CV_32S)
        (total_labels, label_ids, values, centroid) = analysis
        output_mask = np.zeros(self.image.shape, dtype = "uint8") # Mask to remove
        for i in range(1, total_labels): # Check each component
            area = values[i, cv2.CC_STAT_AREA] 
            width = values[i, cv2.CC_STAT_WIDTH]
            height = values[i, cv2.CC_STAT_HEIGHT]

            is_too_small = area < 10
            is_too_big = area > 250
            is_too_wide = width > 3 * height
            do_reject = is_too_small or is_too_big or is_too_wide
            if do_reject:
                component_mask = (label_ids == i).astype("uint8") * 255 # Convert component pixels to 255 to mark white
                output_mask = cv2.bitwise_or(output_mask, component_mask) # Add component to mask
        if self.do_visualize: self.__make_subplot(output_mask, ax, key2, title = "Non-character Components")

        # Subtract mask from image
        self.image = cv2.subtract(self.image, output_mask)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Characters filtered")

    def __process_secondary(self):
        if self.do_visualize:
            key1, key2, key3 = "1", '2', "3"
            fig, ax = self.__make_subplot_figure([key1, key2, key3], title = "3: Secondary processing")

        def remove_lines(do_vertical = True):
            kernel_size = (1, 25)
            if not do_vertical: kernel_size = kernel_size[::-1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            detected_lines = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel, iterations = 2)
            cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(self.image, [c], -1, (0, 0, 0), 2)

        # Remove vertical lines
        remove_lines(True)
        if self.do_visualize: self.__make_subplot(self.image, ax, key1, title = "Remove Vertical Lines")

        # Remove horizontal lines
        remove_lines(False)
        if self.do_visualize: self.__make_subplot(self.image, ax, key2, title = "Remove Horizontal Lines")

    def __filter_sentences(self):
        if self.do_visualize:
            key1, key2, key3 = "1", '2', "3"
            fig, ax = self.__make_subplot_figure([key1, key2, key3], title = "4: Filter Sentences")

        # apply a closing operation using a rectangular kernel to close
        # gaps in between letters
        sentence_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 13))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, sentence_kernel)
        if self.do_visualize: self.__make_subplot(self.image, ax, key1, title = "Join Sentences")

        # Create mask of components that are very unlike sentences
        analysis = cv2.connectedComponentsWithStats(self.image, 4, cv2.CV_32S)
        (total_labels, label_ids, values, centroid) = analysis
        output_mask = np.zeros(self.image.shape, dtype="uint8") # Mask to remove
        for i in range(1, total_labels):  # Check each component
            area = values[i, cv2.CC_STAT_AREA]
            width = values[i, cv2.CC_STAT_WIDTH]
            height = values[i, cv2.CC_STAT_HEIGHT]

            is_too_small = width < 10 or height < 10
            is_not_vertical_box = width > height or area < width * height * 0.3
            do_reject = is_too_small or is_not_vertical_box
            if do_reject:
                component_mask = (label_ids == i).astype("uint8") * 255 # Convert component pixels to 255 to mark white
                output_mask = cv2.bitwise_or(output_mask, component_mask) # Add component to mask
        if self.do_visualize: self.__make_subplot(output_mask, ax, key2, title = "Non-sentence Components")

        self.image = cv2.subtract(self.image, output_mask)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Sentences Filtered")
    
    def __filter_text_blocks(self):
        if self.do_visualize:
            key1, key2, key3 = "1", '2', "3"
            fig, ax = self.__make_subplot_figure([key1, key2, key3], title = "5: Filter Text Blocks")
        
        # apply a closing operation using a rectangular kernel to close
        # gaps between lines of text
        sentence_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 2))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, sentence_kernel)
        if self.do_visualize: self.__make_subplot(self.image, ax, key1, title = "Join Text Blocks")