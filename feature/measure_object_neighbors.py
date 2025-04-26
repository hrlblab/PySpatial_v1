import numpy
import scipy
import skimage
import matplotlib

from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.cpmorphology import strel_disk, centers_of_labels
from centrosome.outline import outline
from cellprofiler_core.constants.measurement import NEIGHBORS

from core.objects import Objects
from utilities.preferences import get_default_colormap


D_ADJACENT = "Adjacent"
D_EXPAND = "Expand until adjacent"
D_WITHIN = "Within a specified distance"

M_NUMBER_OF_NEIGHBORS = "NumberOfNeighbors"
M_PERCENT_TOUCHING = "PercentTouching"
M_FIRST_CLOSEST_OBJECT_NUMBER = "FirstClosestObjectNumber"
M_FIRST_CLOSEST_DISTANCE = "FirstClosestDistance"
M_SECOND_CLOSEST_OBJECT_NUMBER = "SecondClosestObjectNumber"
M_SECOND_CLOSEST_DISTANCE = "SecondClosestDistance"
M_ANGLE_BETWEEN_NEIGHBORS = "AngleBetweenNeighbors"

S_EXPANDED = "Expanded"
S_ADJACENT = "Adjacent"

C_NEIGHBORS = "Neighbors"



class MeasureObjectNeighbors:
    def __init__(self,object,neighbor_object):
        self.object = object
        self.neighbor_object =neighbor_object

        self.distance_method = "Within a specified distance"
        self.distance = 5
        self.wants_count_image = False
        self.count_image_name =None
        self.count_colormap = "Default"
        self.wants_percent_touching_image = False
        self.touching_image_name = "PercentTouching"
        self.touching_colormap = "Default"
        self.wants_excluded_objects = True

        self.neighbors_are_objects=True


    def add_settings(self,distance_method_input,distance_input,wants_count_image_input,count_image_name_input,
                     count_colormap_input,wants_percent_touching_image_input,touching_image_name_input,
                     touching_colormap_input,wants_excluded_objects_input):

        self.distance_method = distance_method_input
        self.distance = distance_input
        self.wants_count_image = wants_count_image_input
        self.count_image_name = count_image_name_input
        self.count_colormap = count_colormap_input
        self.wants_percent_touching_image =wants_percent_touching_image_input
        self.touching_image_name = touching_image_name_input
        self.touching_colormap = touching_colormap_input
        self.wants_excluded_objects = wants_excluded_objects_input

    def run(self):
        objects = self.object
        dimensions = len(objects.shape)
        assert isinstance(objects, Objects)
        has_pixels = objects.areas > 0
        labels = objects.small_removed_segmented
        kept_labels = objects.segmented
        neighbor_objects = self.neighbor_object
        neighbor_labels = neighbor_objects.small_removed_segmented
        neighbor_kept_labels = neighbor_objects.segmented
        assert isinstance(neighbor_objects, Objects)
        if not self.wants_excluded_objects:
            # Remove labels not present in kept segmentation while preserving object IDs.
            mask = neighbor_kept_labels > 0
            neighbor_labels[~mask] = 0
        nobjects = numpy.max(labels)
        nkept_objects = len(objects.indices)
        nneighbors = numpy.max(neighbor_labels)

        _, object_numbers = objects.relate_labels(labels, kept_labels)
        if self.neighbors_are_objects:
            neighbor_numbers = object_numbers
            neighbor_has_pixels = has_pixels
        else:
            _, neighbor_numbers = neighbor_objects.relate_labels(
                neighbor_labels, neighbor_objects.small_removed_segmented
            )
            neighbor_has_pixels = numpy.bincount(neighbor_labels.ravel())[1:] > 0
        neighbor_count = numpy.zeros((nobjects,))
        pixel_count = numpy.zeros((nobjects,))
        first_object_number = numpy.zeros((nobjects,), int)
        second_object_number = numpy.zeros((nobjects,), int)
        first_x_vector = numpy.zeros((nobjects,))
        second_x_vector = numpy.zeros((nobjects,))
        first_y_vector = numpy.zeros((nobjects,))
        second_y_vector = numpy.zeros((nobjects,))
        angle = numpy.zeros((nobjects,))
        percent_touching = numpy.zeros((nobjects,))
        expanded_labels = None
        if self.distance_method == D_EXPAND:
            # Find the i,j coordinates of the nearest foreground point
            # to every background point
            if dimensions == 2:
                i, j = scipy.ndimage.distance_transform_edt(
                    labels == 0, return_distances=False, return_indices=True
                )
                # Assign each background pixel to the label of its nearest
                # foreground pixel. Assign label to label for foreground.
                labels = labels[i, j]
            else:
                k, i, j = scipy.ndimage.distance_transform_edt(
                    labels == 0, return_distances=False, return_indices=True
                )
                labels = labels[k, i, j]
            expanded_labels = labels  # for display
            distance = 1  # dilate once to make touching edges overlap
            scale = S_EXPANDED
            if self.neighbors_are_objects:
                neighbor_labels = labels.copy()
        elif self.distance_method == D_WITHIN:
            distance = self.distance
            scale = str(distance)
        elif self.distance_method == D_ADJACENT:
            distance = 1
            scale = S_ADJACENT
        else:
            raise ValueError("Unknown distance method: %s" % self.distance_method)
        if nneighbors > (1 if self.neighbors_are_objects else 0):
            first_objects = []
            second_objects = []
            object_indexes = numpy.arange(nobjects, dtype=numpy.int32) + 1
            #
            # First, compute the first and second nearest neighbors,
            # and the angles between self and the first and second
            # nearest neighbors
            #
            ocenters = centers_of_labels(objects.small_removed_segmented).transpose()
            ncenters = centers_of_labels(
                neighbor_objects.small_removed_segmented
            ).transpose()
            areas = fix(
                scipy.ndimage.sum(numpy.ones(labels.shape), labels, object_indexes)
            )
            perimeter_outlines = outline(labels)
            perimeters = fix(
                scipy.ndimage.sum(
                    numpy.ones(labels.shape), perimeter_outlines, object_indexes
                )
            )

            #
            # order[:,0] should be arange(nobjects)
            # order[:,1] should be the nearest neighbor
            # order[:,2] should be the next nearest neighbor
            #
            order = numpy.zeros((nobjects, min(nneighbors, 3)), dtype=numpy.uint32)
            j = numpy.arange(nneighbors)
            # (0, 1, 2) unless there are less than 3 neighbors
            partition_keys = tuple(range(min(nneighbors, 3)))
            for i in range(nobjects):
                dr = numpy.sqrt((ocenters[i, 0] - ncenters[j, 0]) ** 2 + (ocenters[i, 1] - ncenters[j, 1]) ** 2)
                order[i, :] = numpy.argpartition(dr, partition_keys)[:3]

            first_neighbor = 1 if self.neighbors_are_objects else 0
            first_object_index = order[:, first_neighbor]
            first_x_vector = ncenters[first_object_index, 1] - ocenters[:, 1]
            first_y_vector = ncenters[first_object_index, 0] - ocenters[:, 0]
            if nneighbors > first_neighbor + 1:
                second_object_index = order[:, first_neighbor + 1]
                second_x_vector = ncenters[second_object_index, 1] - ocenters[:, 1]
                second_y_vector = ncenters[second_object_index, 0] - ocenters[:, 0]
                v1 = numpy.array((first_x_vector, first_y_vector))
                v2 = numpy.array((second_x_vector, second_y_vector))
                #
                # Project the unit vector v1 against the unit vector v2
                #
                dot = numpy.sum(v1 * v2, 0) / numpy.sqrt(
                    numpy.sum(v1 ** 2, 0) * numpy.sum(v2 ** 2, 0)
                )
                angle = numpy.arccos(dot) * 180.0 / numpy.pi

            # Make the structuring element for dilation
            if dimensions == 2:
                strel = strel_disk(distance)
            else:
                strel = skimage.morphology.ball(distance)
            #
            # A little bigger one to enter into the border with a structure
            # that mimics the one used to create the outline
            #
            if dimensions == 2:
                strel_touching = strel_disk(distance + 0.5)
            else:
                strel_touching = skimage.morphology.ball(distance + 0.5)
            #
            # Get the extents for each object and calculate the patch
            # that excises the part of the image that is "distance"
            # away
            if dimensions == 2:
                i, j = numpy.mgrid[0: labels.shape[0], 0: labels.shape[1]]

                minimums_i, maximums_i, _, _ = scipy.ndimage.extrema(
                    i, labels, object_indexes
                )
                minimums_j, maximums_j, _, _ = scipy.ndimage.extrema(
                    j, labels, object_indexes
                )

                minimums_i = numpy.maximum(fix(minimums_i) - distance, 0).astype(int)
                maximums_i = numpy.minimum(
                    fix(maximums_i) + distance + 1, labels.shape[0]
                ).astype(int)
                minimums_j = numpy.maximum(fix(minimums_j) - distance, 0).astype(int)
                maximums_j = numpy.minimum(
                    fix(maximums_j) + distance + 1, labels.shape[1]
                ).astype(int)
            else:
                k, i, j = numpy.mgrid[
                          0: labels.shape[0], 0: labels.shape[1], 0: labels.shape[2]
                          ]

                minimums_k, maximums_k, _, _ = scipy.ndimage.extrema(
                    k, labels, object_indexes
                )
                minimums_i, maximums_i, _, _ = scipy.ndimage.extrema(
                    i, labels, object_indexes
                )
                minimums_j, maximums_j, _, _ = scipy.ndimage.extrema(
                    j, labels, object_indexes
                )

                minimums_k = numpy.maximum(fix(minimums_k) - distance, 0).astype(int)
                maximums_k = numpy.minimum(
                    fix(maximums_k) + distance + 1, labels.shape[0]
                ).astype(int)
                minimums_i = numpy.maximum(fix(minimums_i) - distance, 0).astype(int)
                maximums_i = numpy.minimum(
                    fix(maximums_i) + distance + 1, labels.shape[1]
                ).astype(int)
                minimums_j = numpy.maximum(fix(minimums_j) - distance, 0).astype(int)
                maximums_j = numpy.minimum(
                    fix(maximums_j) + distance + 1, labels.shape[2]
                ).astype(int)
            #
            # Loop over all objects
            # Calculate which ones overlap "index"
            # Calculate how much overlap there is of others to "index"
            #
            for object_number in object_numbers:
                if object_number == 0:
                    #
                    # No corresponding object in small-removed. This means
                    # that the object has no pixels, e.g., not renumbered.
                    #
                    continue
                index = object_number - 1
                if dimensions == 2:
                    patch = labels[
                            minimums_i[index]: maximums_i[index],
                            minimums_j[index]: maximums_j[index],
                            ]
                    npatch = neighbor_labels[
                             minimums_i[index]: maximums_i[index],
                             minimums_j[index]: maximums_j[index],
                             ]
                else:
                    patch = labels[
                            minimums_k[index]: maximums_k[index],
                            minimums_i[index]: maximums_i[index],
                            minimums_j[index]: maximums_j[index],
                            ]
                    npatch = neighbor_labels[
                             minimums_k[index]: maximums_k[index],
                             minimums_i[index]: maximums_i[index],
                             minimums_j[index]: maximums_j[index],
                             ]

                #
                # Find the neighbors
                #
                patch_mask = patch == (index + 1)
                if distance <= 5:
                    extended = scipy.ndimage.binary_dilation(patch_mask, strel)
                else:
                    extended = (
                            scipy.signal.fftconvolve(patch_mask, strel, mode="same") > 0.5
                    )
                neighbors = numpy.unique(npatch[extended])
                neighbors = neighbors[neighbors != 0]
                if self.neighbors_are_objects:
                    neighbors = neighbors[neighbors != object_number]
                nc = len(neighbors)
                neighbor_count[index] = nc
                if nc > 0:
                    first_objects.append(numpy.ones(nc, int) * object_number)
                    second_objects.append(neighbors)
                #
                # Find the # of overlapping pixels. Dilate the neighbors
                # and see how many pixels overlap our image. Use a 3x3
                # structuring element to expand the overlapping edge
                # into the perimeter.
                #
                if dimensions == 2:
                    outline_patch = (
                            perimeter_outlines[
                            minimums_i[index]: maximums_i[index],
                            minimums_j[index]: maximums_j[index],
                            ]
                            == object_number
                    )
                else:
                    outline_patch = (
                            perimeter_outlines[
                            minimums_k[index]: maximums_k[index],
                            minimums_i[index]: maximums_i[index],
                            minimums_j[index]: maximums_j[index],
                            ]
                            == object_number
                    )
                if self.neighbors_are_objects:
                    extendme = (patch != 0) & (patch != object_number)
                    if distance <= 5:
                        extended = scipy.ndimage.binary_dilation(
                            extendme, strel_touching
                        )
                    else:
                        extended = (
                                scipy.signal.fftconvolve(
                                    extendme, strel_touching, mode="same"
                                )
                                > 0.5
                        )
                else:
                    if distance <= 5:
                        extended = scipy.ndimage.binary_dilation(
                            (npatch != 0), strel_touching
                        )
                    else:
                        extended = (
                                scipy.signal.fftconvolve(
                                    (npatch != 0), strel_touching, mode="same"
                                )
                                > 0.5
                        )
                overlap = numpy.sum(outline_patch & extended)
                pixel_count[index] = overlap
            if sum([len(x) for x in first_objects]) > 0:
                first_objects = numpy.hstack(first_objects)
                reverse_object_numbers = numpy.zeros(
                    max(numpy.max(object_numbers), numpy.max(first_objects)) + 1, int
                )
                reverse_object_numbers[object_numbers] = (
                        numpy.arange(len(object_numbers)) + 1
                )
                first_objects = reverse_object_numbers[first_objects]

                second_objects = numpy.hstack(second_objects)
                reverse_neighbor_numbers = numpy.zeros(
                    max(numpy.max(neighbor_numbers), numpy.max(second_objects)) + 1, int
                )
                reverse_neighbor_numbers[neighbor_numbers] = (
                        numpy.arange(len(neighbor_numbers)) + 1
                )
                second_objects = reverse_neighbor_numbers[second_objects]
                to_keep = (first_objects > 0) & (second_objects > 0)
                first_objects = first_objects[to_keep]
                second_objects = second_objects[to_keep]
            else:
                first_objects = numpy.zeros(0, int)
                second_objects = numpy.zeros(0, int)
            percent_touching = pixel_count * 100 / perimeters
            object_indexes = object_numbers - 1
            neighbor_indexes = neighbor_numbers - 1
            #
            # Have to recompute nearest
            #
            first_object_number = numpy.zeros(nkept_objects, int)
            second_object_number = numpy.zeros(nkept_objects, int)
            if nkept_objects > (1 if self.neighbors_are_objects else 0):
                di = (
                        ocenters[object_indexes[:, numpy.newaxis], 0]
                        - ncenters[neighbor_indexes[numpy.newaxis, :], 0]
                )
                dj = (
                        ocenters[object_indexes[:, numpy.newaxis], 1]
                        - ncenters[neighbor_indexes[numpy.newaxis, :], 1]
                )
                distance_matrix = numpy.sqrt(di * di + dj * dj)
                distance_matrix[~has_pixels, :] = numpy.inf
                distance_matrix[:, ~neighbor_has_pixels] = numpy.inf
                #
                # order[:,0] should be arange(nobjects)
                # order[:,1] should be the nearest neighbor
                # order[:,2] should be the next nearest neighbor
                #
                order = numpy.lexsort([distance_matrix]).astype(
                    first_object_number.dtype
                )
                if self.neighbors_are_objects:
                    first_object_number[has_pixels] = order[has_pixels, 1] + 1
                    if nkept_objects > 2:
                        second_object_number[has_pixels] = order[has_pixels, 2] + 1
                else:
                    first_object_number[has_pixels] = order[has_pixels, 0] + 1
                    if order.shape[1] > 1:
                        second_object_number[has_pixels] = order[has_pixels, 1] + 1
        else:
            object_indexes = object_numbers - 1
            neighbor_indexes = neighbor_numbers - 1
            first_objects = numpy.zeros(0, int)
            second_objects = numpy.zeros(0, int)
        #
        # Now convert all measurements from the small-removed to
        # the final number set.
        #
        neighbor_count = neighbor_count[object_indexes]
        neighbor_count[~has_pixels] = 0
        percent_touching = percent_touching[object_indexes]
        percent_touching[~has_pixels] = 0
        first_x_vector = first_x_vector[object_indexes]
        second_x_vector = second_x_vector[object_indexes]
        first_y_vector = first_y_vector[object_indexes]
        second_y_vector = second_y_vector[object_indexes]
        angle = angle[object_indexes]
        #
        # Record the measurements
        #
        # assert isinstance(workspace, Workspace)
        # m = workspace.measurements
        record_measurement=[]

        # assert isinstance(m, Measurements)
        # image_set = workspace.image_set
        features_and_data = [
            (M_NUMBER_OF_NEIGHBORS, neighbor_count),
            (M_FIRST_CLOSEST_OBJECT_NUMBER, first_object_number),
            (
                M_FIRST_CLOSEST_DISTANCE,
                numpy.sqrt(first_x_vector ** 2 + first_y_vector ** 2),
            ),
            (M_SECOND_CLOSEST_OBJECT_NUMBER, second_object_number),
            (
                M_SECOND_CLOSEST_DISTANCE,
                numpy.sqrt(second_x_vector ** 2 + second_y_vector ** 2),
            ),
            (M_ANGLE_BETWEEN_NEIGHBORS, angle),
            (M_PERCENT_TOUCHING, percent_touching),
        ]
        for feature_name, data in features_and_data:
            record_measurement.append([self.get_measurement_name(feature_name), data])
            # m.add_measurement(
            #     self.object_name.value, self.get_measurement_name(feature_name), data
            # )
        # if len(first_objects) > 0:
        #     m.add_relate_measurement(
        #         self.module_num,
        #         NEIGHBORS,
        #         self.object_name.value,
        #         self.object_name.value
        #         if self.neighbors_are_objects
        #         else self.neighbors_name.value,
        #         m.image_set_number * numpy.ones(first_objects.shape, int),
        #         first_objects,
        #         m.image_set_number * numpy.ones(second_objects.shape, int),
        #         second_objects,
        #     )
        #
        # labels = kept_labels
        #
        # neighbor_count_image = numpy.zeros(labels.shape, int)
        # object_mask = objects.segmented != 0
        # object_indexes = objects.segmented[object_mask] - 1
        # neighbor_count_image[object_mask] = neighbor_count[object_indexes]
        # workspace.display_data.neighbor_count_image = neighbor_count_image
        #
        # percent_touching_image = numpy.zeros(labels.shape)
        # percent_touching_image[object_mask] = percent_touching[object_indexes]
        # workspace.display_data.percent_touching_image = percent_touching_image
        #
        # image_set = workspace.image_set
        # if self.wants_count_image:
        #     neighbor_cm_name = self.count_colormap
        #     neighbor_cm = get_colormap(neighbor_cm_name)
        #     sm = matplotlib.cm.ScalarMappable(cmap=neighbor_cm)
        #     img = sm.to_rgba(neighbor_count_image)[:, :, :3]
        #     img[:, :, 0][~object_mask] = 0
        #     img[:, :, 1][~object_mask] = 0
        #     img[:, :, 2][~object_mask] = 0
        #     count_image = Image(img, masking_objects=objects)
        #     image_set.add(self.count_image_name.value, count_image)
        # else:
        #     neighbor_cm_name = "Blues"
        #     neighbor_cm = matplotlib.cm.get_cmap(neighbor_cm_name)
        # if self.wants_percent_touching_image:
        #     percent_touching_cm_name = self.touching_colormap
        #     percent_touching_cm = get_colormap(percent_touching_cm_name)
        #     sm = matplotlib.cm.ScalarMappable(cmap=percent_touching_cm)
        #     img = sm.to_rgba(percent_touching_image)[:, :, :3]
        #     img[:, :, 0][~object_mask] = 0
        #     img[:, :, 1][~object_mask] = 0
        #     img[:, :, 2][~object_mask] = 0
        #     touching_image = Image(img, masking_objects=objects)
        #     image_set.add(self.touching_image_name, touching_image)
        # else:
        #     percent_touching_cm_name = "Oranges"
        #     percent_touching_cm = matplotlib.cm.get_cmap(percent_touching_cm_name)
        #
        # if self.show_window:
        #     workspace.display_data.neighbor_cm_name = neighbor_cm_name
        #     workspace.display_data.percent_touching_cm_name = percent_touching_cm_name
        #     workspace.display_data.orig_labels = objects.segmented
        #     workspace.display_data.neighbor_labels = neighbor_labels
        #     workspace.display_data.expanded_labels = expanded_labels
        #     workspace.display_data.object_mask = object_mask
        #     workspace.display_data.dimensions = dimensions
        return record_measurement

    def get_measurement_name(self, feature):
        if self.distance_method == D_EXPAND:
            scale = S_EXPANDED
        elif self.distance_method == D_WITHIN:
            scale = str(self.distance)
        elif self.distance_method == D_ADJACENT:
            scale = S_ADJACENT
        if self.neighbors_are_objects:
            return "_".join((C_NEIGHBORS, feature, scale))
        else:
            return "_".join((C_NEIGHBORS, feature, self.neighbors_name.value, scale))


def get_colormap(name):
    """Get colormap, accounting for possible request for default"""
    if name == "Default":
        name = get_default_colormap()
    return matplotlib.cm.get_cmap(name)


