from dataclasses import dataclass
import cv2

# Class for handling connected component analysis and making component data
# available in a more convenient format.
class ComponentAnalyzer:
    @property
    def n_labels(self) -> int:
        return len(self.components)

    def __init__(self, image):
        analysis = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)
        total_labels, label_ids, values, centroid = analysis

        self.components: list[ComponentData] = []
        for i in range(1, total_labels): # Check each component
            im_height, im_width = image.shape

            data = ComponentData(
            label = i,
            y1 = values[i, cv2.CC_STAT_TOP],
            x1 = values[i, cv2.CC_STAT_LEFT],
            height = values[i, cv2.CC_STAT_HEIGHT],
            width = values[i, cv2.CC_STAT_WIDTH],
            area = values[i, cv2.CC_STAT_AREA],
            boolean_mesh = label_ids == i,
            centroid = centroid,
            image_height = im_height,
            image_width = im_width)

            self.components.append(data)

    def find_segments(self, image, buffer = 0):
        segments = []
        for comp in self.components:
            [y1, y2, x1, x2] = comp.calc_buffer_coords(buffer)
            segments.append(image[y1:y2, x1:x2])
        return segments
        

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
    image_height: int
    image_width: int

    @property
    def y2(self) -> int:
        return self.y1 + self.height - 1
    
    @property
    def x2(self) -> int:
        return self.x1 + self.width - 1

    @property
    def bounding_area(self) -> int:
        return self.height * self.width

    # Create buffer around component while respecting image limits
    def calc_buffer_coords(self, buffer):
        y1 = max(0, self.y1 - buffer)
        x1 = max(0, self.x1 - buffer)

        y2 = min(self.image_height - 1, self.y2 + buffer)
        x2 = min(self.image_width - 1, self.x2 + buffer)

        return y1, y2, x1, x2

        
        