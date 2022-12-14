from dataclasses import dataclass
import cv2

# Class for handling connected component analysis and making component data
# available in a more convenient format.
class ComponentAnalyzer:
    def __init__(self, image):
        analysis = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)
        total_labels, label_ids, values, centroid = analysis

        self.components: list = []
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
    def buffer_coords(self, buffer):
        self.y2 = min(self.image_height - 1, self.y2 + buffer)
        self.x2 = min(self.image_width - 1, self.x2 + buffer)
        self.x1 = max(0, self.x1 - buffer)
        self.y1 = max(0, self.y1 - buffer)