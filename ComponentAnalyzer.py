import math
from dataclasses import dataclass

import cv2
import numpy as np


# Class for handling connected component analysis and making component data
# available in a more convenient format.
class ComponentAnalyzer:
    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def image_height(self) -> int:
        return self.image_shape[0]

    @property
    def image_width(self) -> int:
        return self.image_shape[1]

    def __init__(self, image):
        # self.image_shape: tuple = image.shape
        self.image_shape = (len(image), len(image[0]))

        analysis = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)
        total_labels, label_ids, values, centroid = analysis

        self.components: list[ComponentData] = []
        for i in range(1, total_labels):  # Check each component

            data = ComponentData(
                label=i,
                y1=values[i, cv2.CC_STAT_TOP],
                x1=values[i, cv2.CC_STAT_LEFT],
                height=values[i, cv2.CC_STAT_HEIGHT],
                width=values[i, cv2.CC_STAT_WIDTH],
                area=values[i, cv2.CC_STAT_AREA],
                boolean_mesh=label_ids == i,
                centroid=centroid,
                image_shape=self.image_shape,
            )

            self.components.append(data)

    def find_segments(self, image, buffer=0, return_coordinates=False):
        # Since the segments were found in an image width uniform width,
        # if the input image has different width from the original image, the coordinates are transformed.
        # That way this function can be used for any version of the image scaled by width.
        im_height = image.shape[0]
        im_width = image.shape[1]
        image_ratio = im_width / self.image_width

        segments = []
        y1s = []
        x1s = []
        y2s = []
        x2s = []
        for comp in self.components:
            [y1, y2, x1, x2] = comp.calc_buffer_coords(buffer)
            y1 = max(0, math.floor(y1 * image_ratio))
            x1 = max(0, math.floor(x1 * image_ratio))
            y2 = min(im_height - 1, math.ceil(y2 * image_ratio))
            x2 = min(im_width - 1, math.ceil(x2 * image_ratio))

            segments.append(image[y1:y2, x1:x2])
            y1s.append(y1)
            x1s.append(x1)
            y2s.append(y2)
            x2s.append(x2)

        if return_coordinates:
            return segments, y1s, x1s, y2s, x2s
        else:
            return segments

    # Create a mask consisting of all components fulfilling the criteria of a test function taking
    # a ComponentData object as input.
    # If no test function is provided, creates a mask of all components.
    def create_mask(self, test_function=None):
        output_mask = np.zeros(self.image_shape, dtype="uint8")  # Mask to remove
        for comp in self.components:
            if test_function == None or test_function(comp):
                component_mask = (
                    comp.boolean_mesh.astype("uint8") * 255
                )  # Convert component pixels to 255 to mark white
                output_mask = cv2.bitwise_or(
                    output_mask, component_mask
                )  # Add component to mask
        return output_mask


# Data for each component.
@dataclass
class ComponentData:
    label: int
    y1: int
    x1: int
    height: int
    width: int
    area: int
    boolean_mesh: list
    centroid: tuple
    image_shape: tuple

    # Derived component properties:
    @property
    def y2(self) -> int:
        return self.y1 + self.height - 1

    @property
    def x2(self) -> int:
        return self.x1 + self.width - 1

    @property
    def bounding_area(self) -> int:
        return self.height * self.width

    @property
    def image_height(self) -> int:
        return self.image_shape[0]

    @property
    def image_width(self) -> int:
        return self.image_shape[1]

    @property
    def image_area(self) -> int:
        return self.image_height * self.image_width

    # Indicates if component contains the image edge:
    @property
    def is_top_edge(self) -> int:
        return self.y1 == 0

    @property
    def is_left_edge(self) -> int:
        return self.x1 == 0

    @property
    def is_bottom_edge(self) -> int:
        return self.y2 == self.image_height - 1

    @property
    def is_right_edge(self) -> int:
        return self.x2 == self.image_width - 1

    # Find buffer around component while respecting image limits:
    def calc_buffer_coords(self, buffer):
        y1 = max(0, self.y1 - buffer)
        x1 = max(0, self.x1 - buffer)

        y2 = min(self.image_height - 1, self.y2 + buffer)
        x2 = min(self.image_width - 1, self.x2 + buffer)

        return y1, y2, x1, x2
