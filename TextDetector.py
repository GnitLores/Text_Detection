import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import timeit
from ComponentAnalyzer import *
import pytesseract


class TextDetector:
    def __init__(
        self,
        image,
        is_text_vertical=True,
        show_process=False,
        show_segments=False,
        show_result=False,
        do_profile=False,
    ):
        self.original_image = image
        self.vert_text = is_text_vertical
        self.show_process = show_process
        self.show_segments = show_segments
        self.show_result = show_result
        self.do_profile = do_profile
        # self.height, self.width, _ = image.shape

        self.page_width = 600
        self.original_page_width = self.original_image.shape[1]
        self.uniform_image_ratio = self.original_page_width / self.page_width

        self.dpi = 96
        self.fig_width_wide = 1600
        self.fig_height_wide = 1000
        self.fig_width_tall = 800
        self.fig_height_tall = 1200

    # Function block that executes each step of the image processing algorithm and optionally
    # times and profiles each step.
    def detect_text(self):
        if self.do_profile:
            t0 = timeit.default_timer()
        self.__preprocess()
        if self.do_profile:
            t1 = timeit.default_timer()
        self.__filter_chars()
        if self.do_profile:
            t2 = timeit.default_timer()
        self.__process_secondary()
        if self.do_profile:
            t3 = timeit.default_timer()
        self.__filter_sentences()
        if self.do_profile:
            t4 = timeit.default_timer()
        self.__filter_text_blocks()
        if self.do_profile:
            t5 = timeit.default_timer()
        self.__select_text_areas()
        if self.do_profile:
            t6 = timeit.default_timer()

        if self.do_profile:
            width, precision = 3, 2
            print_time = lambda text, t_a, t_b: print(
                f"{text}: {t_b - t_a:{width}.{precision}} s"
            )
            print_time("Preprocessing", t0, t1)
            print_time("Filter chars", t1, t2)
            print_time("Secondary processing", t2, t3)
            print_time("Filter sentences", t3, t4)
            print_time("Filter text blocks", t4, t5)
            print_time("Select text areas", t5, t6)

    # Creates a subplot figure for displaying the image.
    # subplot_keys determines the dimensions and keys of the subplot.
    def __make_subplot_figure(self, subplot_keys, title=""):
        fig, ax = plt.subplot_mosaic(
            [subplot_keys],
            figsize=(self.fig_width_wide / self.dpi, self.fig_height_wide / self.dpi),
            dpi=self.dpi,
        )
        fig.suptitle(title)
        return fig, ax

    # Creates a figure with a subplot grid for displaying a variable amount of small segments.
    # Keys are be automatically distributed and the dimensions are approximately square, growing by column first.
    def __make_subplot_grid_figure(self, n_subplots, title=""):
        n_cols = math.ceil(math.sqrt(n_subplots))
        n_rows = math.ceil(n_subplots / n_cols)
        subplot_keys = [
            [c + (n_cols * r) + 1 for c in range(n_cols)] for r in range(n_rows)
        ]
        fig, ax = plt.subplot_mosaic(
            subplot_keys,
            figsize=(self.fig_width_wide / self.dpi, self.fig_height_wide / self.dpi),
            dpi=self.dpi,
        )
        fig.suptitle(title)
        return fig, ax

    # Display an image in a subplot axis.
    def __make_subplot(self, image, ax, key, colormap="gray", title=""):
        ax[key].imshow(image, cmap=colormap)
        ax[key].set_title(title)
        ax[key].axis("off")

    # Plot a list of small image segments in a grid of subplots.
    def __plot_segments(self, segments, title="", descriptions=None):
        if descriptions == None:
            descriptions = ["" for _ in range(len(segments))]
        fig, ax = self.__make_subplot_grid_figure(len(segments), title)

        for i, seg in enumerate(segments):
            self.__make_subplot(seg, ax, i + 1, title=f"{i + 1}: " + descriptions[i])

    def __make_subplot_graph(self, data, ax, key, title=""):
        ax[key].plot(data)
        ax[key].set_title(title)
        ax[key].axis("off")

    # Initial processing:
    # Downscale image to uniform width.
    # Blur to reduce noise.
    # Emphasize black on white object.
    # Threshold image.
    def __preprocess(self):

        if self.show_process:
            key1, key2, key3 = "1", "2", "3"
            fig, ax = self.__make_subplot_figure(
                [key1, key2, key3], title="1: Preprocessing"
            )
        if self.show_process:
            self.__make_subplot(self.original_image, ax, key1, title="Original Image")

        # Resize to uniform width
        def resize_to_uniform_width(im, new_width, inter=cv2.INTER_AREA):
            old_height, old_width = im.shape[0], im.shape[1]
            ratio = new_width / float(old_width)
            dimensions = (new_width, int(old_height * ratio))
            return cv2.resize(im, dimensions, interpolation=inter)

        # self.image = imutils.resize(self.original_image, width=self.page_width)
        self.image = resize_to_uniform_width(
            self.original_image, new_width=self.page_width
        )

        # Convert to grayscale
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # smooth the image using Gaussian to reduce high frequeny noise
        gauss_size = self.page_width // 200
        if gauss_size % 2 == 0:
            gauss_size += 1  # Must be odd
        self.image = cv2.GaussianBlur(self.image, (gauss_size, gauss_size), 0)
        if self.show_process:
            self.__make_subplot(
                self.image, ax, key2, title="Rescaled, Greyscale, Blurred image"
            )

        # Blackhat - enhances dark objects of interest in a bright background.
        # The black-hat transform is defined as the difference between the closing and the input image.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.page_width // 120, self.page_width // 50)
        )
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_BLACKHAT, kernel)
        self.preprocessed_image = self.image.copy()
        if self.show_process:
            self.__make_subplot(self.image, ax, key3, title="Blackhat Transform")

        # Get black/white image with otsu threshold
        self.image = cv2.threshold(
            self.image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]
        self.binary_image = self.image.copy()
        if self.show_process:
            self.__make_subplot(self.image, ax, key1, title="Thresholded image")

    # Remove everything that looks very different from a character:
    # Do a connected component analysis.
    def __filter_chars(self):

        if self.show_process:
            key1, key2, key3 = "1", "2", "3"
            fig, ax = self.__make_subplot_figure(
                [key1, key2, key3], title="2: Filter Characters"
            )

        # Create mask of components that are very unlike characters
        def filter_component(comp: ComponentData):
            is_too_small = comp.area < self.page_width // 60
            is_too_big = comp.area > self.page_width // 2.4
            is_too_wide = comp.width > 3 * comp.height
            do_reject = is_too_small or is_too_big or is_too_wide
            return do_reject

        analyzer = ComponentAnalyzer(self.image)
        mask = analyzer.create_mask(filter_component)
        if self.show_process:
            self.__make_subplot(mask, ax, key2, title="Non-character Components")

        # Subtract mask from image
        self.image = cv2.subtract(self.image, mask)
        if self.show_process:
            self.__make_subplot(self.image, ax, key3, title="Characters filtered")

    # Secondary processing to remove small artifacts that can remain.
    # Remove thin line fragments.
    def __process_secondary(self):
        if self.show_process:
            key1, key2, key3 = "1", "2", "3"
            fig, ax = self.__make_subplot_figure(
                [key1, key2, key3], title="3: Secondary processing"
            )

        def remove_lines(do_vertical=True):
            kernel_size = (max(1, self.page_width // 600), self.page_width // 24)
            if not do_vertical:
                kernel_size = kernel_size[::-1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            detected_lines = cv2.morphologyEx(
                self.image, cv2.MORPH_OPEN, kernel, iterations=2
            )
            cnts = cv2.findContours(
                detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(self.image, [c], -1, (0, 0, 0), 2)

        # Remove vertical lines
        remove_lines(True)
        if self.show_process:
            self.__make_subplot(self.image, ax, key1, title="Remove Vertical Lines")

        # Remove horizontal lines
        remove_lines(False)
        if self.show_process:
            self.__make_subplot(self.image, ax, key2, title="Remove Horizontal Lines")

        self.image_after_secondary_processing = self.image.copy()

    # Remove everything that looks very different from a sentence:
    # Join character into sentences.
    # Do a connected component analysis.
    def __filter_sentences(self):
        if self.show_process:
            key1, key2, key3 = "1", "2", "3"
            fig, ax = self.__make_subplot_figure(
                [key1, key2, key3], title="4: Filter Sentences"
            )

        # apply a closing operation using a rectangular kernel to close
        # gaps in between letters
        sentence_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.page_width // 300, self.page_width // 50)
        )
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, sentence_kernel)
        if self.show_process:
            self.__make_subplot(self.image, ax, key1, title="Join Sentences")

        # Create mask of components that are very unlike sentences
        def filter_component(comp: ComponentData):
            is_too_small = (
                comp.width < self.page_width // 60
                or comp.height < self.page_width // 60
            )
            is_not_vertical_box = (
                comp.width > comp.height or comp.area < comp.bounding_area * 0.3
            )
            do_reject = is_too_small or is_not_vertical_box
            return do_reject

        analyzer = ComponentAnalyzer(self.image)
        mask = analyzer.create_mask(filter_component)
        if self.show_process:
            self.__make_subplot(mask, ax, key2, title="Non-sentence Components")

        self.image = cv2.subtract(self.image, mask)
        if self.show_process:
            self.__make_subplot(self.image, ax, key3, title="Sentences Filtered")

        self.sentence_image = self.image.copy()

    # Remove everything that looks very different from a text block (a connected component of one or more sentences):
    # Join sentences into text blocks.
    # Do a connected component analysis.
    def __filter_text_blocks(self):
        if self.show_process:
            key1, key2, key3 = "1", "2", "3"
            fig, ax = self.__make_subplot_figure(
                [key1, key2, key3], title="5: Filter Text Blocks"
            )

        # apply a closing operation using a rectangular kernel to close
        # gaps between lines of text
        sentence_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.page_width // 50, self.page_width // 300)
        )
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, sentence_kernel)
        if self.show_process:
            self.__make_subplot(self.image, ax, key1, title="Join Text Blocks")

        # Create mask of components that are very unlike text blocks
        def filter_component(comp: ComponentData):
            is_too_small = comp.area < self.page_width // 2.4
            do_reject = is_too_small
            return do_reject

        analyzer = ComponentAnalyzer(self.image)
        mask = analyzer.create_mask(filter_component)
        if self.show_process:
            self.__make_subplot(mask, ax, key2, title="Non-Text Block Components")

        self.image = cv2.subtract(self.image, mask)
        if self.show_process:
            self.__make_subplot(self.image, ax, key3, title="Text Blocks Filtered")

    # Select text areas to use for OCR based on discriminating text blocks.
    def __select_text_areas(self):
        analyzer = ComponentAnalyzer(self.image)

        buffer = self.page_width // 200
        segments, y1s, x1s, y2s, x2s = analyzer.find_segments(
            self.original_image, buffer=buffer, return_coordinates=True
        )

        if self.show_segments:
            self.__plot_segments(segments, title="Original")

        for i, seg in enumerate(segments):
            seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
            segments[i] = cv2.threshold(
                seg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )[1]
            segments[i] = cv2.bitwise_not(segments[i])

        if self.show_segments:
            self.__plot_segments(segments, title="Thresholded")

        # img = self.preprocessed_image
        # # img = self.binary_image

        # if self.show_segments: self.__plot_segments(analyzer.find_segments(self.image, buffer = buffer), title = "Text Area Candidates (text block components)")
        # if self.show_segments: self.__plot_segments(analyzer.find_segments(img, buffer = buffer), title = "Text Area Candidates (base image)")

        # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # # img = cv2.bitwise_not(img)
        # segments = analyzer.find_segments(img, buffer = buffer)
        # if self.show_segments: self.__plot_segments(segments, title = "Text Area Candidates (thresholded)")

        def remove_grain_components(segment):
            def filter_component(c: ComponentData):
                return c.area < self.original_page_width // 100

            analyzer = ComponentAnalyzer(segment)
            mask = analyzer.create_mask(filter_component)
            return cv2.subtract(segment, mask)

        for i, seg in enumerate(segments):
            segments[i] = remove_grain_components(seg)
        if self.show_segments:
            self.__plot_segments(segments, title="Grain Removed")

        def remove_border_components(segment):
            def filter_component(c: ComponentData):
                return (
                    c.is_left_edge
                    or c.is_right_edge
                    or c.is_top_edge
                    or c.is_bottom_edge
                )

            analyzer = ComponentAnalyzer(segment)
            mask = analyzer.create_mask(filter_component)
            return cv2.subtract(segment, mask)

        for i, seg in enumerate(segments):
            segments[i] = remove_border_components(seg)
        if self.show_segments:
            self.__plot_segments(segments, title="Border Components Removed")

        def remove_large_components(segment):
            def filter_component(c: ComponentData):
                return c.height > c.image_height * 0.5 or c.area > c.image_area * 0.3

            analyzer = ComponentAnalyzer(segment)
            mask = analyzer.create_mask(filter_component)
            return cv2.subtract(segment, mask)

        for i, seg in enumerate(segments):
            segments[i] = remove_large_components(seg)
        if self.show_segments:
            self.__plot_segments(segments, title="Large Components Removed")

        # def remove_non_text_components(segment):
        #     def filter_component(c: ComponentData):
        #         # Filter oddly shaped compnents above a certain size threshold (to avoid catching small character components)
        #         size_threshold = round((self.page_width // 40) * self.uniform_image_ratio)
        #         too_wide = c.width > size_threshold and c.width > c.height * 2
        #         too_narrow = c.height > size_threshold and c.height > c.width * 3 # A bit more conservative to prevent catching two characters in same sentence joined
        #         return too_wide or too_narrow

        #     analyzer = ComponentAnalyzer(segment)
        #     mask = analyzer.create_mask(filter_component)
        #     return cv2.subtract(segment, mask)

        # for i, seg in enumerate(segments):
        #     segments[i] = remove_non_text_components(seg)
        # if self.show_segments: self.__plot_segments(segments, title = "Non-Text Components Removed")

        includes = [True] * len(segments)

        square_ratio_cutoff = 0.5
        filled_square_ratio_cutoff = 0.3
        squariness = 0.7
        filledness = 0.4
        kernel_width = self.original_page_width // 250
        kernel_height = 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_width, kernel_height)
        )
        joined_segments = []
        params = []
        for i, seg in enumerate(segments):
            joined_segment = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)
            joined_segments.append(joined_segment)

            analyzer = ComponentAnalyzer(joined_segment)
            n_square = 0
            n_filled = 0
            n_filled_square = 0
            for c in analyzer.components:
                is_square = (
                    c.width > c.height * squariness and c.height > c.width * squariness
                )
                is_filled = c.area > c.height * c.width * filledness
                if is_square:
                    n_square += 1
                if is_filled:
                    n_filled += 1
                if is_filled and is_square:
                    n_filled_square += 1
            n_segments = len(analyzer.components)
            square_ratio = n_square / n_segments if n_square > 0 else 0
            filled_ratio = n_filled / n_segments if n_filled > 0 else 0
            filled_square_ratio = (
                n_filled_square / n_segments if n_filled_square > 0 else 0
            )

            includes[i] = (
                includes[i]
                and square_ratio > square_ratio_cutoff
                and filled_square_ratio > filled_square_ratio_cutoff
            )

            params.append(
                f"square={square_ratio:.2f}, filled={filled_ratio:.2f}, filled_square={filled_square_ratio:.2f}"
            )
        if self.show_segments:
            self.__plot_segments(
                joined_segments, title="Square Characters", descriptions=params
            )

        # ratios = []
        # for i, seg in enumerate(segments):
        #     height_ratio = seg.shape[0] / self.original_image.shape[0]
        #     width_ratio = seg.shape[1] / self.original_image.shape[1]
        #     area_ratio = (seg.shape[0] * seg.shape[1]) / (self.original_image.shape[0] * self.original_image.shape[1])
        #     ratios.append(f'a_ration={area_ratio:.3f}, h_ratio={height_ratio:.2f}, w_ratio={width_ratio:.2f}')
        # self.__plot_segments(segments, title = "Discrimination:", descriptions = ratios)

        descriptions = []
        for i, seg in enumerate(segments):
            row_sum = np.sum(seg, axis=1) // 255
            max_intens = max(row_sum)
            zero_fraction = (
                sum([(s < 0.05 * max_intens) or s == 0 for s in row_sum]) / len(row_sum)
            ) * 100

            filling_ratio = (np.sum(seg) // 255) / np.prod(seg.shape) * 100

            segment_size_ratio = (
                np.prod(seg.shape) / np.prod(self.original_image.shape) * 100
            )

            includes[i] = (
                includes[i]
                and filling_ratio > 10
                and zero_fraction < 50
                and segment_size_ratio > 0.08
            )
            description = f"Fill={filling_ratio:.1f}, Zeros={zero_fraction:.1f},Size={segment_size_ratio:.2f}"
            descriptions.append(description)

        if self.show_segments:
            self.__plot_segments(
                segments, title="Discrimination:", descriptions=descriptions
            )

        if self.show_result:
            # self.__plot_segments(analyzer.find_segments(self.original_image, buffer = buffer), title = "Discrimination:", descriptions = descriptions)

            # _, y1s, x1s, y2s, x2s = analyzer.find_segments(self.original_image, buffer = buffer, return_coordinates = True)
            fig, ax = self.__make_subplot_figure(
                ["all", "included"], title="Readable Areas"
            )
            # rgb_img = cv2.cvtColor(binary_img, cv.CV_GRAY2RGB)
            self.__make_subplot(
                self.original_image, ax, "all", colormap="gray", title="Candidate Areas"
            )
            self.__make_subplot(
                self.original_image,
                ax,
                "included",
                colormap="gray",
                title="Included Areas",
            )
            buffer = 2  # Move left and upper border a bit to display rectangle nicely
            for i in range(len(x1s)):
                # Create a Rectangle patch
                width = x2s[i] - x1s[i] - 1 + buffer
                height = y2s[i] - y1s[i] - 1 + buffer
                rect_red_all = patches.Rectangle(
                    (x1s[i] - buffer, y1s[i] - buffer),
                    width,
                    height,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                rect_red_include = patches.Rectangle(
                    (x1s[i] - buffer, y1s[i] - buffer),
                    width,
                    height,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                rect_blue_all = patches.Rectangle(
                    (x1s[i] - buffer, y1s[i] - buffer),
                    width,
                    height,
                    linewidth=1,
                    edgecolor="b",
                    facecolor="none",
                )

                if includes[i]:
                    ax["included"].add_patch(rect_red_include)
                    ax["all"].add_patch(rect_red_all)
                else:
                    ax["all"].add_patch(rect_blue_all)

        def map_segment_coordinates_to_original_image():
            self.uniform_image_ratio

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
        # return

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
