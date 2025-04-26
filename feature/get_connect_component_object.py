import cv2

import core.objects
import numpy as np
import scipy

class GetConnectComponentObject:
    size_range = (float("-inf"), float("inf"))
    exclude_size = False

    def __init__(self):
        pass


    def get_connect_label(self,uint8_image):
        #gray_uint8_img = cv2.cvtColor(uint8_image, cv2.COLOR_BGR2GRAY)
        gray_uint8_img=uint8_image.copy()
        gray_uint8_img = cv2.threshold(gray_uint8_img, 0, 255, cv2.THRESH_OTSU)[1]
        num_labels, labels_im = cv2.connectedComponents(gray_uint8_img, connectivity=8)

        return num_labels, labels_im

    def run(self,uint8_image,image_instance):

        num_labels, labeled_image = self.get_connect_label(uint8_image)
        object_count=num_labels-1
        unedited_labels = labeled_image.copy()


        # Filter out small and large objects
        size_excluded_labeled_image = labeled_image.copy()
        labeled_image, small_removed_labels = self.filter_on_size(
            labeled_image, object_count
        )
        size_excluded_labeled_image[labeled_image > 0] = 0


        objects = core.objects.Objects()
        objects.segmented = labeled_image
        objects.unedited_segmented = unedited_labels
        objects.small_removed_segmented =small_removed_labels
        objects.parent_image = image_instance

        return objects


    def filter_on_size(self, labeled_image, object_count):
        """ Filter the labeled image based on the size range

        labeled_image - pixel image labels
        object_count - # of objects in the labeled image
        returns the labeled image, and the labeled image with the
        small objects removed
        """
        if self.exclude_size and object_count > 0:
            areas = scipy.ndimage.measurements.sum(
                np.ones(labeled_image.shape),
                labeled_image,
                np.array(list(range(0, object_count + 1)), dtype=np.int32),
            )
            areas = np.array(areas, dtype=int)
            min_allowed_area = (
                    np.pi * (min(self.size_range) * min(self.size_range)) / 4
            )
            max_allowed_area = (
                    np.pi * (max(self.size_range) * max(self.size_range)) / 4
            )
            # area_image has the area of the object at every pixel within the object
            area_image = areas[labeled_image]
            labeled_image[area_image < min_allowed_area] = 0
            small_removed_labels = labeled_image.copy()
            labeled_image[area_image > max_allowed_area] = 0
        else:
            small_removed_labels = labeled_image.copy()
        return labeled_image, small_removed_labels
