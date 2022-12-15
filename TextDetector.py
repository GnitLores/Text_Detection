import math
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import timeit
from ComponentAnalyzer import *
import pytesseract
from PIL import Image

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
        self.__select_text_areas()
        if self.do_profile: t6 = timeit.default_timer()

        if self.do_profile:
            width, precision = 3, 2
            print_time = lambda text, t_a, t_b : print(f'{text}: {t_b - t_a:{width}.{precision}} s')
            print_time("Preprocessing", t0, t1)
            print_time("Filter chars", t1, t2)
            print_time("Secondary processing", t2, t3)
            print_time("Filter sentences", t3, t4)
            print_time("Filter text blocks", t4, t5)
            print_time("Select text areas", t5, t6)

        if self.do_visualize: plt.show()
        # plt.show() # TMP REMOVE!!

    def __make_subplot_figure(self, subplot_keys, title = ""):
        dpi = 96
        figure_width = 1600
        figure_height = 1000
        fig, ax = plt.subplot_mosaic([subplot_keys], figsize = (figure_width / dpi, figure_height / dpi), dpi = dpi)
        fig.suptitle(title)
        return fig, ax

    def __make_subplot_grid_figure(self, n_subplots, title = ""):
        n_cols = math.ceil(math.sqrt(n_subplots))
        n_rows = math.ceil(n_subplots / n_cols)
        subplot_keys = [[c + (n_cols * r) + 1 for c in range(n_cols)] for r in range(n_rows)]
        dpi = 96
        figure_width = 1600
        figure_height = 1000
        fig, ax = plt.subplot_mosaic(subplot_keys, figsize = (figure_width / dpi, figure_height / dpi), dpi = dpi)
        fig.suptitle(title)
        return fig, ax

    def __make_subplot(self, image, ax, key, colormap = "gray", title = ""):
        ax[key].imshow(image, cmap = colormap)
        ax[key].set_title(title)
        ax[key].axis('off')

    def __make_subplot_graph(self, data, ax, key, title = ""):
        ax[key].plot(data)
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
        if gauss_size % 2 == 0: gauss_size += 1 # Must be odd
        self.image = cv2.GaussianBlur(self.image, (gauss_size, gauss_size), 0)
        if self.do_visualize: self.__make_subplot(self.image, ax, key2, title = "Rescaled, Greyscale, Blurred image")
        self.resized_image = self.image.copy()

        # Blackhat - enhances dark objects of interest in a bright background.
        # The black-hat transform is defined as the difference between the closing and the input image.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.page_width // 120, self.page_width // 50))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_BLACKHAT, kernel)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Blackhat Transform")

    def __filter_chars(self):
        
        if self.do_visualize:
            key1, key2, key3 = "1", '2', "3"
            fig, ax = self.__make_subplot_figure([key1, key2, key3], title = "2: Filter Characters")

        # Get black/white image with otsu threshold
        self.binary_image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.image = self.binary_image.copy()
        if self.do_visualize: self.__make_subplot(self.image, ax, key1, title = "Thresholded image")

        # Create mask of components that are very unlike characters
        def filter_component(comp: ComponentData):
            is_too_small = comp.area < self.page_width // 60
            is_too_big = comp.area > self.page_width // 2.4
            is_too_wide = comp.width > 3 * comp.height
            do_reject = is_too_small or is_too_big or is_too_wide
            return do_reject

        analyzer = ComponentAnalyzer(self.image)
        mask = analyzer.create_mask(filter_component)
        if self.do_visualize: self.__make_subplot(mask, ax, key2, title = "Non-character Components")

        # Subtract mask from image
        self.image = cv2.subtract(self.image, mask)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Characters filtered")

    def __process_secondary(self):
        if self.do_visualize:
            key1, key2, key3 = "1", '2', "3"
            fig, ax = self.__make_subplot_figure([key1, key2, key3], title = "3: Secondary processing")

        def remove_lines(do_vertical = True):
            kernel_size = (max(1, self.page_width // 600), self.page_width // 24)
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

        self.image_after_secondary_processing = self.image.copy()

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
        def filter_component(comp: ComponentData):
            is_too_small = comp.width < self.page_width // 60 or comp.height < self.page_width // 60
            is_not_vertical_box = comp.width > comp.height or comp.area < comp.bounding_area * 0.3
            do_reject = is_too_small or is_not_vertical_box
            return do_reject

        analyzer = ComponentAnalyzer(self.image)
        mask = analyzer.create_mask(filter_component)
        if self.do_visualize: self.__make_subplot(mask, ax, key2, title = "Non-sentence Components")

        self.image = cv2.subtract(self.image, mask)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Sentences Filtered")

        self.sentence_image = self.image.copy()
    
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
        def filter_component(comp: ComponentData):
            is_too_small = comp.area < self.page_width // 2.4
            do_reject = is_too_small
            return do_reject

        analyzer = ComponentAnalyzer(self.image)
        mask = analyzer.create_mask(filter_component)
        if self.do_visualize: self.__make_subplot(mask, ax, key2, title = "Non-Text Block Components")

        self.image = cv2.subtract(self.image, mask)
        if self.do_visualize: self.__make_subplot(self.image, ax, key3, title = "Text Blocks Filtered")

    def __plot_segments(self, segments, title = "", descriptions = None):
        if descriptions == None:
            descriptions = ["" for _ in range(len(segments))]
        fig, ax = self.__make_subplot_grid_figure(len(segments), title)

        for i, seg in enumerate(segments):
            self.__make_subplot(seg, ax, i + 1, title = f'{i + 1}: ' + descriptions[i])

    def __select_text_areas(self):
        visualize_segments = False
        # visualize_segments = True
        analyzer = ComponentAnalyzer(self.image)

        img = self.resized_image
        buffer = self.page_width // 200

        if visualize_segments: self.__plot_segments(analyzer.find_segments(self.image, buffer = buffer), title = "Text Area Candidates (text block components)")
        if visualize_segments: self.__plot_segments(analyzer.find_segments(img, buffer = buffer), title = "Text Area Candidates (base image)")

        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        img = cv2.bitwise_not(img)
        segments = analyzer.find_segments(img, buffer = buffer)
        if visualize_segments: self.__plot_segments(segments, title = "Text Area Candidates (thresholded)")

        def remove_border_components(segment):
            def filter_component(c: ComponentData):
                return c.is_left_edge or c.is_right_edge or c.is_top_edge or c.is_bottom_edge
                    
            analyzer = ComponentAnalyzer(segment)
            mask = analyzer.create_mask(filter_component)
            return cv2.subtract(segment, mask)

        # dilation_kernel = np.ones((2, 2), np.uint8)
        for i, seg in enumerate(segments):
            # segments[i] = cv2.dilate(segments[i], kernel, iterations=1)
            segments[i] = remove_border_components(seg)
        if visualize_segments: self.__plot_segments(segments, title = "Border Components Removed")


        # def remove_non_text_components(segment):
        #     def filter_component(c: ComponentData):
        #         too_thin_and_wide = c.width > c.height * 3
        #         too_small = c.area < self.page_width // 100
        #         return too_thin_and_wide or too_small
                    
        #     analyzer = ComponentAnalyzer(segment)
        #     mask = analyzer.create_mask(filter_component)
        #     return cv2.subtract(segment, mask)

        # for i, seg in enumerate(segments):
        #     segments[i] = remove_non_text_components(seg)
        # if visualize_segments: self.__plot_segments(segments, title = "Non-text components Removed")

        # sentence_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.page_width // 300, self.page_width // 50))
        # for i, seg in enumerate(segments):
        #     segments[i] = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, sentence_kernel)
        # if visualize_segments: self.__plot_segments(segments, title = "Sentences Joined")

        _, original_page_width, _ = self.original_image.shape
        image_ratio = original_page_width / self.page_width
        y1s = []
        x1s = []
        y2s = []
        x2s = []
        # heights = []
        # widths = []
        descriptions = []
        for i, seg in enumerate(segments):
            row_sum = np.sum(seg, axis = 1) // 255
            max_intens = max(row_sum)
            zero_fraction = sum([(s < 0.05 * max_intens) or s == 0 for s in row_sum]) / len(row_sum)

            filling_ratio = (np.sum(seg) // 255) / np.prod(seg.shape)

            exclude = filling_ratio < 0.1 or zero_fraction > 0.5
            description = f'Fill={filling_ratio:.2f}, Zeros={zero_fraction:.2f}'
            descriptions.append(description)

            if not exclude:
                x1s.append(math.floor(analyzer.components[i].x1 * image_ratio))
                y1s.append(math.floor(analyzer.components[i].y1 * image_ratio))
                x2s.append(math.ceil(analyzer.components[i].x2 * image_ratio))
                y2s.append(math.ceil(analyzer.components[i].y2 * image_ratio))
                # heights.append(analyzer.components[i].height)
                # widths.append(analyzer.components[i].width)
        if visualize_segments: self.__plot_segments(segments, title = "Discrimination:", descriptions = descriptions)

        fig, ax = plt.subplots()
        # rgb_img = cv2.cvtColor(binary_img, cv.CV_GRAY2RGB)
        ax.imshow(self.original_image)
        buffer = 5
        for i in range(len(x1s)):
            # Create a Rectangle patch
            width = x2s[i] - x1s[i] - 1 + buffer * 2
            height = y2s[i] - y1s[i] - 1 + buffer * 2
            rect = patches.Rectangle((x1s[i] - buffer, y1s[i] - buffer), width, height, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
        pass
        



        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.page_width // 150, self.page_width // 600))
        # for i in range(1, total_labels): # Check each component
        #     # Calculate coordinates with small buffer
        #     x1 = values[i, cv2.CC_STAT_LEFT]
        #     y1 = values[i, cv2.CC_STAT_TOP]
        #     w = values[i, cv2.CC_STAT_WIDTH]
        #     h = values[i, cv2.CC_STAT_HEIGHT]
        #     im_h, im_w = tmp_img.shape
            
        #     y2 = min(im_h - 1, y1 + h + buffer)
        #     x2 = min(im_w - 1, x1 + w + buffer)
        #     x1 = max(0, x1 - buffer)
        #     y1 = max(0, y1 - buffer)

        #     component_image = tmp_img[y1:y2, x1:x2]
        #     do_include, description = self.__analyse_candidate(component_image)
        #     image_section = self.resized_image[y1:y2, x1:x2]
        #     processed_section = cv2.threshold(image_section, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #     processed_section = cv2.bitwise_not(processed_section)
        #     # kernel = np.ones((2, 2), np.uint8)
        #     # # kernel = np.ones((4, 1), np.uint8)
        #     # processed_section = cv2.dilate(processed_section, kernel, iterations=1)
        #     self.__make_subplot(component_image, ax1, i, title = f'{i}: ' + description)
        #     self.__make_subplot(image_section, ax2, i, title = f'{i}: ' + description)
        #     self.__make_subplot(processed_section, ax3, i, title = f'{i}: ' + description)
        return

    # def __analyse_candidate(self, candidate_image):
    #         total_labels, label_ids, values, centroid = self.__find_component_stats(candidate_image)
    #         n_components_total = total_labels - 1
    #         widths = []
    #         heights = []
    #         areas = []
    #         is_squares = []
    #         for i in range(1, total_labels): # Check each component
    #             width = values[i, cv2.CC_STAT_WIDTH]
    #             height = values[i, cv2.CC_STAT_HEIGHT]
    #             area = values[i, cv2.CC_STAT_AREA]
    #             if area < 20: continue
                
    #             is_square = 0.5 < width / height < 2

    #             widths.append(width)
    #             heights.append(height)
    #             areas.append(area)
    #             is_squares.append(is_square)
    #         n_square = sum(is_squares)
    #         m_height = np.mean(heights)
    #         m_width = np.mean(widths)
    #         mean_size = m_height + m_width / 2
    #         description = f'{n_components_total} total, {len(widths)} comps'
    #         description = ""
    #         return True, description 