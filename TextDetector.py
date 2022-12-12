import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
import timeit

class TextDetector:
    def __init__(self, image, do_visualize = False, do_profile = False, is_text_vertical = True):
        self.original_image = image
        self.do_visualize = do_visualize
        self.do_profile = do_profile
        self.vert_text = is_text_vertical
        self.height, self.width, _ = image.shape

        self.page_width = 600
        
    def detect_text(self):
        if self.do_profile: t0 = timeit.default_timer()
        self.__preprocess()
        if self.do_profile: t1 = timeit.default_timer()
        self.__filter_chars()
        if self.do_profile: t2 = timeit.default_timer()
        self.__process_secondary()
        if self.do_profile: t3 = timeit.default_timer()
        self.__filter_sentences()
        if self.do_profile: t4 = timeit.default_timer()
        self.__filter_text_blocks()
        if self.do_profile: t5 = timeit.default_timer()

        if self.do_profile:
            width, precision = 3, 2
            print_time = lambda text, t_a, t_b : print(f'{text}: {t_b - t_a:{width}.{precision}} s')
            print_time("Preprocessing", t0, t1)
            print_time("Filter chars", t1, t2)
            print_time("Secondary processing", t2, t3)
            print_time("Filter sentences", t3, t4)
            print_time("Filter text blocks", t4, t5)

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
        self.image = imutils.resize(self.original_image, width = self.page_width)

        # Convert to grayscale
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # smooth the image using Gaussian to reduce high frequeny noise
        gauss_size = self.page_width // 200
        self.image = cv2.GaussianBlur(self.image, (gauss_size, gauss_size), 0)
        if self.do_visualize: self.__make_subplot(self.image, ax, key2, title = "Rescaled, Greyscale, Blurred image")

        # Blackhat - enhances dark objects of interest in a bright background.
        # The black-hat transform is defined as the difference between the closing and the input image.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.page_width // 120, self.page_width // 50))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_BLACKHAT, kernel)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Blackhat Transform")
    
    def __analyze_connected_components(self, filter_component):
        analysis = cv2.connectedComponentsWithStats(self.image, 4, cv2.CV_32S)
        (total_labels, label_ids, values, centroid) = analysis
        output_mask = np.zeros(self.image.shape, dtype = "uint8") # Mask to remove
        for i in range(1, total_labels): # Check each component
            area = values[i, cv2.CC_STAT_AREA] 
            width = values[i, cv2.CC_STAT_WIDTH]
            height = values[i, cv2.CC_STAT_HEIGHT]

            if filter_component(area, width, height):
                component_mask = (label_ids == i).astype("uint8") * 255 # Convert component pixels to 255 to mark white
                output_mask = cv2.bitwise_or(output_mask, component_mask) # Add component to mask
        return output_mask

    def __filter_chars(self):
        
        if self.do_visualize:
            key1, key2, key3 = "1", '2', "3"
            fig, ax = self.__make_subplot_figure([key1, key2, key3], title = "2: Filter Characters")

        # Get black/white image with otsu threshold
        self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if self.do_visualize: self.__make_subplot(self.image, ax, key1, title = "Thresholded image")

        # Create mask of components that are very unlike characters
        def filter_component(area, width, height):
            is_too_small = area < self.page_width // 60
            is_too_big = area > self.page_width // 2.4
            is_too_wide = width > 3 * height
            do_reject = is_too_small or is_too_big or is_too_wide
            return do_reject

        mask = self.__analyze_connected_components(filter_component)
        if self.do_visualize: self.__make_subplot(mask, ax, key2, title = "Non-character Components")

        # Subtract mask from image
        self.image = cv2.subtract(self.image, mask)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Characters filtered")

    def __process_secondary(self):
        if self.do_visualize:
            key1, key2, key3 = "1", '2', "3"
            fig, ax = self.__make_subplot_figure([key1, key2, key3], title = "3: Secondary processing")

        def remove_lines(do_vertical = True):
            kernel_size = (self.page_width // 600, self.page_width // 24)
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
        sentence_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.page_width // 300, self.page_width // 50))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, sentence_kernel)
        if self.do_visualize: self.__make_subplot(self.image, ax, key1, title = "Join Sentences")

        # Create mask of components that are very unlike sentences
        def filter_component(area, width, height):
            is_too_small = width < self.page_width // 60 or height < self.page_width // 60
            is_not_vertical_box = width > height or area < width * height * 0.3
            do_reject = is_too_small or is_not_vertical_box
            return do_reject

        mask = self.__analyze_connected_components(filter_component)
        if self.do_visualize: self.__make_subplot(mask, ax, key2, title = "Non-sentence Components")

        self.image = cv2.subtract(self.image, mask)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Sentences Filtered")
    
    def __filter_text_blocks(self):
        if self.do_visualize:
            key1, key2, key3 = "1", '2', "3"
            fig, ax = self.__make_subplot_figure([key1, key2, key3], title = "5: Filter Text Blocks")
        
        # apply a closing operation using a rectangular kernel to close
        # gaps between lines of text
        sentence_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.page_width // 50, self.page_width // 300))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, sentence_kernel)
        if self.do_visualize: self.__make_subplot(self.image, ax, key1, title = "Join Text Blocks")

        # Create mask of components that are very unlike text blocks
        def filter_component(area, width, height):
            is_too_small = area < self.page_width // 2.4
            do_reject = is_too_small
            return do_reject

        mask = self.__analyze_connected_components(filter_component)
        if self.do_visualize: self.__make_subplot(mask, ax, key2, title = "Non-Text Block Components")

        self.image = cv2.subtract(self.image, mask)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Text Blocks Filtered")