from utilities.threshold_new import Threshold
import centrosome
import numpy as np
import scipy
import skimage as ski
import math

import core.objects



UN_INTENSITY = "Intensity"
UN_SHAPE = "Shape"
UN_LOG = "Laplacian of Gaussian"
UN_NONE = "None"

WA_INTENSITY = "Intensity"
WA_SHAPE = "Shape"
WA_PROPAGATE = "Propagate"
WA_NONE = "None"

LIMIT_NONE = "Continue"
LIMIT_TRUNCATE = "Truncate"
LIMIT_ERASE = "Erase"

DEFAULT_MAXIMA_COLOR = "Blue"

"""Never fill holes"""
FH_NEVER = "Never"
FH_THRESHOLDING = "After both thresholding and declumping"
FH_DECLUMP = "After declumping only"

FH_ALL = (FH_NEVER, FH_THRESHOLDING, FH_DECLUMP)

# Settings text which is referenced in various places in the help
SIZE_RANGE_SETTING_TEXT = "Typical diameter of objects, in pixel units (Min,Max)"
EXCLUDE_SIZE_SETTING_TEXT = "Discard objects outside the diameter range?"
AUTOMATIC_SMOOTHING_SETTING_TEXT = (
    "Automatically calculate size of smoothing filter for declumping?"
)
SMOOTHING_FILTER_SIZE_SETTING_TEXT = "Size of smoothing filter"
AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT = (
    "Automatically calculate minimum allowed distance between local maxima?"
)






class IdentifyPrimaryObject:
    # parameter
   # threshold_scope = "Adaptive"
   # local_operation = "otsu"
    #two_class_otsu = "Two_class"
    #threshold_smoothing_scale = 0.5
    #threshold_correction_factor = 1.0
    #threshold_range = (0, 1)
    #adaptive_window_size = 50
    #log_transform = False
    # threshold method=ostu

    advanced = True
    basic = (not advanced)  # advanced
    size_range = (10, 200)
    fill_holes = "FH_THRESHOLDING"

    unclump_method = "UN_SHAPE"  # "Method to distinguish clumped objects"
    watershed_method = "WA_INTENSITY"  # "Method to draw dividing lines between clumped objects"
    low_res_maxima = True
    automatic_suppression = True
    automatic_smoothing = True
    exclude_border_objects = False
    exclude_size = True
    limit_choice = "LIMIT_NONE"
    automatic = False
    show_window=True

    FH_THRESHOLDING='After both thresholding and declumping'

    # functions:
    def __init__(self):
        self.threshold = Threshold()
        self.use_advanced=False # "Use advanced settings?"
        self.size_range=(10, 40) #"Typical diameter of objects, in pixel units (Min,Max)"
        self.exclude_size=True #"Discard objects outside the diameter range?"
        self.exclude_border_objects=True #"Discard objects touching the border of the image?"
        # advance setting
        self.unclump_method="Shape" #"Method to distinguish clumped objects"
        self.watershed_method="Intensity" #"Method to draw dividing lines between clumped objects"
        self.smoothing_filter_size=10 #*(Used only when distinguishing between clumped objects)*
        self.maxima_suppression_size=7 #"Suppress local maxima that are closer than this minimum allowed distance"
        self.low_res_maxima=None #"Speed up by using lower-resolution image to find local maxima?"
        self.fill_holes='After both thresholding and declumping' #"Fill holes in identified objects?"
        self.automatic_smoothing=None # "Automatically calculate size of smoothing filter for declumping?"
        self.automatic_suppression=True # "Automatically calculate minimum allowed distance between local maxima?"
        self.limit_choice="Continue" #"Handling of objects if excessive number of objects identified"
        self.maximum_object_count=None #"Maximum number of objects"

        self.threshold.add_setting("Global","Minimum Cross-Entropy","Minimum Cross-Entropy",
                                   1.3488,1,(0,1),0,
                                   None,"Two classes","Foreground",0.05,
                                   0.05,"Mean","Standard deviation",2,50,False,"Otsu")

        self.volumetric=False #3D?
        #self.automatic=False


    def add_setting(self,use_advanced_input,size_range_input,exclude_size_input,exclude_border_objects_input
                    ,unclump_method_input,watershed_method_input,smoothing_filter_size_input,maxima_suppression_size_input,
                    low_res_maxima_input,fill_holes_input,automatic_smoothing_input,automatic_suppression_input,
                    limit_choice_input,maximum_object_count_input):
        self.use_advanced = use_advanced_input  # "Use advanced settings?"
        self.size_range = size_range_input  # "Typical diameter of objects, in pixel units (Min,Max)"
        self.exclude_size = exclude_size_input  # "Discard objects outside the diameter range?"
        self.exclude_border_objects = exclude_border_objects_input  # "Discard objects touching the border of the image?"
        # advance setting
        self.unclump_method = unclump_method_input  # "Method to distinguish clumped objects"
        self.watershed_method = watershed_method_input  # "Method to draw dividing lines between clumped objects"
        self.smoothing_filter_size = smoothing_filter_size_input  # *(Used only when distinguishing between clumped objects)*
        self.maxima_suppression_size = maxima_suppression_size_input  # "Suppress local maxima that are closer than this minimum allowed distance"
        self.low_res_maxima = low_res_maxima_input  # "Speed up by using lower-resolution image to find local maxima?"
        self.fill_holes = fill_holes_input  # "Fill holes in identified objects?"
        self.automatic_smoothing = automatic_smoothing_input  # "Automatically calculate size of smoothing filter for declumping?"
        self.automatic_suppression = automatic_suppression_input  # "Automatically calculate minimum allowed distance between local maxima?"
        self.limit_choice = limit_choice_input  # "Handling of objects if excessive number of objects identified"
        self.maximum_object_count = maximum_object_count_input  # "Maximum number of objects"

    def add_threshold_settings(self,threshold_scope_input,global_operation_input,local_operation_input,threshold_smoothing_scale_input,
                    threshold_correction_factor_input,threshold_range_input,manual_threshold_input,thresholding_measurement_input,
                    two_class_otsu_input,assign_middle_to_foreground_input,lower_outlier_fraction_input,upper_outlier_fraction_input,
                    averaging_method_input,variance_method_input,number_of_deviations_input,adaptive_window_size_input,log_transform_input,threshold_operation_input):
        self.threshold.add_setting(threshold_scope_input,global_operation_input,local_operation_input,threshold_smoothing_scale_input,
                    threshold_correction_factor_input,threshold_range_input,manual_threshold_input,thresholding_measurement_input,
                    two_class_otsu_input,assign_middle_to_foreground_input,lower_outlier_fraction_input,upper_outlier_fraction_input,
                    averaging_method_input,variance_method_input,number_of_deviations_input,adaptive_window_size_input,log_transform_input,threshold_operation_input)


    def run(self,image,mask_img=None):
        self.image = image.pixel_data
        if mask_img==None:
            self.mask = image.mask
        else:
            self.mask = mask_img

        binary_image, global_threshold, sigma = self._threshold_image(
        )

        #
        # Fill background holes inside foreground objects
        #
        def size_fn(size, is_foreground):
            return size < min(self.size_range) * max(self.size_range)

        if self.basic or self.fill_holes == FH_THRESHOLDING:
            binary_image = centrosome.cpmorphology.fill_labeled_holes(binary_image, size_fn=size_fn)

        labeled_image, object_count = scipy.ndimage.label(binary_image, np.ones((3, 3), bool))

        (labeled_image, object_count, maxima_suppression_size,) = self.separate_neighboring_objects(self.image,
                                                                                                    self.mask,
                                                                                                    labeled_image,
                                                                                                    object_count)

        unedited_labels = labeled_image.copy()

        # Filter out objects touching the border or mask
        border_excluded_labeled_image = labeled_image.copy()
        labeled_image = self.filter_on_border(image, labeled_image)
        border_excluded_labeled_image[labeled_image > 0] = 0

        # Filter out small and large objects
        size_excluded_labeled_image = labeled_image.copy()
        labeled_image, small_removed_labels = self.filter_on_size(
            labeled_image, object_count
        )
        size_excluded_labeled_image[labeled_image > 0] = 0

        #
        # Fill holes again after watershed
        #
        if self.basic or self.fill_holes != FH_NEVER:
            labeled_image = centrosome.cpmorphology.fill_labeled_holes(labeled_image)

        # Relabel the image
        labeled_image, object_count = centrosome.cpmorphology.relabel(labeled_image)

        if self.advanced and self.limit_choice == LIMIT_ERASE:
            if object_count > self.maximum_object_count.value:
                labeled_image = np.zeros(labeled_image.shape, int)
                border_excluded_labeled_image = np.zeros(labeled_image.shape, int)
                size_excluded_labeled_image = np.zeros(labeled_image.shape, int)
                object_count = 0

        # Make an outline image
        outline_image = centrosome.outline.outline(labeled_image)
        outline_size_excluded_image = centrosome.outline.outline(
            size_excluded_labeled_image
        )
        outline_border_excluded_image = centrosome.outline.outline(
            border_excluded_labeled_image
        )

        if self.show_window:
            statistics = []
            statistics.append(["# of accepted objects", "%d" % object_count])
            if object_count > 0:
                areas = scipy.ndimage.sum(
                    np.ones(labeled_image.shape),
                    labeled_image,
                    np.arange(1, object_count + 1),
                )
                areas.sort()
                low_diameter = (
                        math.sqrt(float(areas[object_count // 10]) / np.pi) * 2
                )
                median_diameter = (
                        math.sqrt(float(areas[object_count // 2]) / np.pi) * 2
                )
                high_diameter = (
                        math.sqrt(float(areas[object_count * 9 // 10]) / np.pi) * 2
                )
                statistics.append(
                    ["10th pctile diameter", "%.1f pixels" % low_diameter]
                )
                statistics.append(["Median diameter", "%.1f pixels" % median_diameter])
                statistics.append(
                    ["90th pctile diameter", "%.1f pixels" % high_diameter]
                )
                object_area = np.sum(areas)
                total_area = np.product(labeled_image.shape[:2])
                statistics.append(
                    [
                        "Area covered by objects",
                        "%.1f %%" % (100.0 * float(object_area) / float(total_area)),
                    ]
                )
                statistics.append(["Thresholding filter size", "%.1f" % sigma])
                statistics.append(["Threshold", "%0.3g" % global_threshold])
                if self.basic or self.unclump_method != UN_NONE:
                    statistics.append(
                        [
                            "Declumping smoothing filter size",
                            "%.1f" % (self.calc_smoothing_filter_size()),
                        ]
                    )
                    statistics.append(
                        ["Maxima suppression size", "%.1f" % maxima_suppression_size]
                    )
            else:
                statistics.append(["Threshold", "%0.3g" % global_threshold])

        #########################

        # # Add image measurements
        # objname = self.y_name.value
        # measurements = workspace.measurements

        # # Add label matrices to the object set
        objects = core.objects.Objects()
        objects.segmented = labeled_image
        objects.unedited_segmented = unedited_labels
        objects.small_removed_segmented = small_removed_labels
        objects.parent_image = image

        # workspace.object_set.add_objects(objects, self.y_name.value)

        return objects




    def _threshold_image(self):
        final_threshold, orig_threshold, guide_threshold = self.threshold.get_threshold(
            image=self.image,mask=self.mask,volumetric=self.volumetric,automatic=self.automatic
        )

        self.threshold.add_threshold_measurements(
            final_threshold, orig_threshold, guide_threshold
        )

        binary_image, sigma = self.threshold.apply_threshold(
            self.image, self.mask, final_threshold, self.automatic
        )

        self.threshold.add_fg_bg_measurements(
            self.image, self.mask, binary_image
        )

        return binary_image, np.mean(np.atleast_1d(final_threshold)), sigma




    def separate_neighboring_objects(self, input_image, input_mask, labeled_image, object_count):
        """Separate objects based on local maxima or distance transform

        workspace - get the image from here

        labeled_image - image labeled by scipy.ndimage.label

        object_count  - # of objects in image

        returns revised labeled_image, object count, maxima_suppression_size,
        LoG threshold and filter diameter
        """
        if self.advanced and (
                self.unclump_method == UN_NONE or self.watershed_method == WA_NONE
        ):
            return labeled_image, object_count, 7

        image = input_image
        mask = input_mask

        blurred_image = self.smooth_image(image, mask)
        if min(self.size_range) > 10 and (self.basic or self.low_res_maxima):
            image_resize_factor = 10.0 / float(min(self.size_range))
            if self.basic or self.automatic_suppression:
                maxima_suppression_size = 7
            else:
                maxima_suppression_size = (
                        self.maxima_suppression_size * image_resize_factor + 0.5
                )
            reported_maxima_suppression_size = (
                    maxima_suppression_size / image_resize_factor
            )
        else:
            image_resize_factor = 1.0
            if self.basic or self.automatic_suppression:
                maxima_suppression_size = min(self.size_range) / 1.5
            else:
                maxima_suppression_size = self.maxima_suppression_size
            reported_maxima_suppression_size = maxima_suppression_size
        maxima_mask = centrosome.cpmorphology.strel_disk(
            max(1, maxima_suppression_size - 0.5)
        )
        distance_transformed_image = None
        if self.basic or self.unclump_method == UN_INTENSITY:
            # Remove dim maxima
            maxima_image = self.get_maxima(
                blurred_image, labeled_image, maxima_mask, image_resize_factor
            )
        elif self.unclump_method == UN_SHAPE:
            if self.fill_holes == FH_NEVER:
                # For shape, even if the user doesn't want to fill holes,
                # a point far away from the edge might be near a hole.
                # So we fill just for this part.
                foreground = (
                        centrosome.cpmorphology.fill_labeled_holes(labeled_image) > 0
                )
            else:
                foreground = labeled_image > 0
            distance_transformed_image = scipy.ndimage.distance_transform_edt(
                foreground
            )
            # randomize the distance slightly to get unique maxima
            np.random.seed(0)
            distance_transformed_image += np.random.uniform(
                0, 0.001, distance_transformed_image.shape
            )
            maxima_image = self.get_maxima(
                distance_transformed_image,
                labeled_image,
                maxima_mask,
                image_resize_factor,
            )
        else:
            raise ValueError(
                "Unsupported local maxima method: %s" % self.unclump_method
            )

        # Create the image for watershed
        if self.basic or self.watershed_method == WA_INTENSITY:
            # use the reverse of the image to get valleys at peaks
            watershed_image = 1 - image
        elif self.watershed_method == WA_SHAPE:
            if distance_transformed_image is None:
                distance_transformed_image = scipy.ndimage.distance_transform_edt(
                    labeled_image > 0
                )
            watershed_image = -distance_transformed_image
            watershed_image = watershed_image - np.min(watershed_image)
        elif self.watershed_method == WA_PROPAGATE:
            # No image used
            pass
        else:
            raise NotImplementedError(
                "Watershed method %s is not implemented" % self.watershed_method
            )
        #
        # Create a marker array where the unlabeled image has a label of
        # -(nobjects+1)
        # and every local maximum has a unique label which will become
        # the object's label. The labels are negative because that
        # makes the watershed algorithm use FIFO for the pixels which
        # yields fair boundaries when markers compete for pixels.
        #
        self.labeled_maxima, object_count = scipy.ndimage.label(
            maxima_image, np.ones((3, 3), bool)
        )
        if self.advanced and self.watershed_method == WA_PROPAGATE:
            watershed_boundaries, distance = centrosome.propagate.propagate(
                np.zeros(self.labeled_maxima.shape),
                self.labeled_maxima,
                labeled_image != 0,
                1.0,
            )
        else:
            markers_dtype = (
                np.int16
                if object_count < np.iinfo(np.int16).max
                else np.int32
            )
            markers = np.zeros(watershed_image.shape, markers_dtype)
            markers[self.labeled_maxima > 0] = -self.labeled_maxima[
                self.labeled_maxima > 0
                ]

            #
            # Some labels have only one maker in them, some have multiple and
            # will be split up.
            #

            watershed_boundaries = ski.segmentation.watershed(
                connectivity=np.ones((3, 3), bool),
                image=watershed_image,
                markers=markers,
                mask=labeled_image != 0,
            )

            watershed_boundaries = -watershed_boundaries

        return watershed_boundaries, object_count, reported_maxima_suppression_size

    def smooth_image(self, image, mask):
        """Apply the smoothing filter to the image"""

        filter_size = self.calc_smoothing_filter_size()
        if filter_size == 0:
            return image
        sigma = filter_size / 2.35
        #
        # We not only want to smooth using a Gaussian, but we want to limit
        # the spread of the smoothing to 2 SD, partly to make things happen
        # locally, partly to make things run faster, partly to try to match
        # the Matlab behavior.
        #
        filter_size = max(int(float(filter_size) / 2.0), 1)
        f = (
                1
                / np.sqrt(2.0 * np.pi)
                / sigma
                * np.exp(
            -0.5 * np.arange(-filter_size, filter_size + 1) ** 2 / sigma ** 2
        )
        )

        def fgaussian(image):
            output = scipy.ndimage.convolve1d(image, f, axis=0, mode="constant")
            return scipy.ndimage.convolve1d(output, f, axis=1, mode="constant")

        #
        # Use the trick where you similarly convolve an array of ones to find
        # out the edge effects, then divide to correct the edge effects
        #
        edge_array = fgaussian(mask.astype(float))
        masked_image = image.copy()
        masked_image[~mask] = 0
        smoothed_image = fgaussian(masked_image)
        masked_image[mask] = smoothed_image[mask] / edge_array[mask]
        return masked_image

    def calc_smoothing_filter_size(self):
        """Return the size of the smoothing filter, calculating it if in automatic mode"""
        if self.automatic_smoothing:
            return 2.35 * min(self.size_range) / 3.5
        else:
            return self.smoothing_filter_size

    def get_maxima(self, image, labeled_image, maxima_mask, image_resize_factor):
        if image_resize_factor < 1.0:
            shape = np.array(image.shape) * image_resize_factor
            i_j = (
                    np.mgrid[0: shape[0], 0: shape[1]].astype(float)
                    / image_resize_factor
            )
            resized_image = scipy.ndimage.map_coordinates(image, i_j)
            resized_labels = scipy.ndimage.map_coordinates(
                labeled_image, i_j, order=0
            ).astype(labeled_image.dtype)

        else:
            resized_image = image
            resized_labels = labeled_image
        #
        # find local maxima
        #
        if maxima_mask is not None:
            binary_maxima_image = centrosome.cpmorphology.is_local_maximum(
                resized_image, resized_labels, maxima_mask
            )
            binary_maxima_image[resized_image <= 0] = 0
        else:
            binary_maxima_image = (resized_image > 0) & (labeled_image > 0)
        if image_resize_factor < 1.0:
            inverse_resize_factor = float(image.shape[0]) / float(
                binary_maxima_image.shape[0]
            )
            i_j = (
                    np.mgrid[0: image.shape[0], 0: image.shape[1]].astype(float)
                    / inverse_resize_factor
            )
            binary_maxima_image = (
                    scipy.ndimage.map_coordinates(binary_maxima_image.astype(float), i_j)
                    > 0.5
            )
            assert binary_maxima_image.shape[0] == image.shape[0]
            assert binary_maxima_image.shape[1] == image.shape[1]

        # Erode blobs of touching maxima to a single point

        shrunk_image = centrosome.cpmorphology.binary_shrink(binary_maxima_image)
        return shrunk_image

    def filter_on_border(self, image, labeled_image):
        """Filter out objects touching the border

        In addition, if the image has a mask, filter out objects
        touching the border of the mask.
        """
        if self.exclude_border_objects:
            border_labels = list(labeled_image[0, :])
            border_labels.extend(labeled_image[:, 0])
            border_labels.extend(labeled_image[labeled_image.shape[0] - 1, :])
            border_labels.extend(labeled_image[:, labeled_image.shape[1] - 1])
            border_labels = np.array(border_labels)
            #
            # the following histogram has a value > 0 for any object
            # with a border pixel
            #
            histogram = scipy.sparse.coo_matrix(
                (
                    np.ones(border_labels.shape),
                    (border_labels, np.zeros(border_labels.shape)),
                ),
                shape=(np.max(labeled_image) + 1, 1),
            ).todense()
            histogram = np.array(histogram).flatten()
            if any(histogram[1:] > 0):
                histogram_image = histogram[labeled_image]
                labeled_image[histogram_image > 0] = 0
            elif image.has_mask:
                # The assumption here is that, if nothing touches the border,
                # the mask is a large, elliptical mask that tells you where the
                # well is. That's the way the old Matlab code works and it's duplicated here
                #
                # The operation below gets the mask pixels that are on the border of the mask
                # The erosion turns all pixels touching an edge to zero. The not of this
                # is the border + formerly masked-out pixels.
                mask_border = np.logical_not(
                    scipy.ndimage.binary_erosion(image.mask)
                )
                mask_border = np.logical_and(mask_border, image.mask)
                border_labels = labeled_image[mask_border]
                border_labels = border_labels.flatten()
                histogram = scipy.sparse.coo_matrix(
                    (
                        np.ones(border_labels.shape),
                        (border_labels, np.zeros(border_labels.shape)),
                    ),
                    shape=(np.max(labeled_image) + 1, 1),
                ).todense()
                histogram = np.array(histogram).flatten()
                if any(histogram[1:] > 0):
                    histogram_image = histogram[labeled_image]
                    labeled_image[histogram_image > 0] = 0
        return labeled_image

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



