from utilities.object import crop_labels_and_image
import numpy as np
import centrosome
import centrosome.propagate
import scipy

from utilities.measure_object_intensity_distribution_setting import MeasureObjectIntensityDistributionSetting


C_SELF = "These objects"
C_EDGES_OF_OTHER = "Edges of other objects"

M_CATEGORY = "RadialDistribution"
F_FRAC_AT_D = "FracAtD"
F_MEAN_FRAC = "MeanFrac"
F_RADIAL_CV = "RadialCV"

FF_SCALE = "%dof%d"
FF_OVERFLOW = "Overflow"
FF_GENERIC = "_%s_" + FF_SCALE
FF_FRAC_AT_D = F_FRAC_AT_D + FF_GENERIC
FF_MEAN_FRAC = F_MEAN_FRAC + FF_GENERIC
FF_RADIAL_CV = F_RADIAL_CV + FF_GENERIC

MF_FRAC_AT_D = "_".join((M_CATEGORY, FF_FRAC_AT_D))
MF_MEAN_FRAC = "_".join((M_CATEGORY, FF_MEAN_FRAC))
MF_RADIAL_CV = "_".join((M_CATEGORY, FF_RADIAL_CV))
OF_FRAC_AT_D = "_".join((M_CATEGORY, F_FRAC_AT_D, "%s", FF_OVERFLOW))
OF_MEAN_FRAC = "_".join((M_CATEGORY, F_MEAN_FRAC, "%s", FF_OVERFLOW))
OF_RADIAL_CV = "_".join((M_CATEGORY, F_RADIAL_CV, "%s", FF_OVERFLOW))

Z_MAGNITUDES_AND_PHASE = "Magnitudes and phase"
FF_ZERNIKE_MAGNITUDE = "ZernikeMagnitude"
FF_ZERNIKE_PHASE = "ZernikePhase"


class MeasureObjectIntensityDistribution:

    def __init__(self):
        self.setting_ls=[]
        self.center_choice="These objects"
        self.wants_zernikes="Magnitudes and phase"
        self.zernike_degree=9
        self.image_name='podocyte_nuclei'
        print("MeasureObjectIntensityDistribution")


    def add_setting(self,bin_count_input,can_remove_input,maximum_radius_input,wants_scaled_input):
        temp_setting=MeasureObjectIntensityDistributionSetting()
        temp_setting.bin_count=bin_count_input
        temp_setting.can_remove=can_remove_input
        temp_setting.maximum_radius=maximum_radius_input
        temp_setting.wants_scaled=wants_scaled_input
        temp_setting.settings=[wants_scaled_input,bin_count_input,maximum_radius_input]
        self.setting_ls.append(temp_setting)


    def run(self,image,objects_ls):
        d = {}
        intensity_distribution_result = []
        for objects in objects_ls:
            # add if object=None
            if objects==None or len(objects.areas)==0:
                return None


            for bin_count_settings in self.setting_ls:
                intensity_distribution_result.extend(
                    self.do_measurements(
                        image,
                        objects,
                        None,
                        bin_count_settings,
                        d,))

            if self.wants_zernikes != "None":
                self.calculate_zernikes(intensity_distribution_result,objects,image)

        return intensity_distribution_result


    def calculate_zernikes(self,intensity_distribution_result,objects,image):
        zernike_indexes = centrosome.zernike.get_zernike_indexes(
            self.zernike_degree + 1
        )

        #meas = workspace.measurements

        # for o in self.objects:
        #     object_name = o.object_name.value

        #objects = workspace.object_set.get_objects(object_name)

        #
        # First, get a table of centers and radii of minimum enclosing
        # circles per object
        #
        ij = np.zeros((objects.count + 1, 2))

        r = np.zeros(objects.count + 1)

        for labels, indexes in objects.get_labels():
            ij_, r_ = centrosome.cpmorphology.minimum_enclosing_circle(
                labels, indexes
            )

            ij[indexes] = ij_

            r[indexes] = r_

        #
        # Then compute x and y, the position of each labeled pixel
        # within a unit circle around the object
        #
        ijv = objects.ijv

        l = ijv[:, 2]

        yx = (ijv[:, :2] - ij[l, :]) / r[l, np.newaxis]

        z = centrosome.zernike.construct_zernike_polynomials(
            yx[:, 1], yx[:, 0], zernike_indexes
        )

        # for image_name in self.images_list.value:
        #     image = workspace.image_set.get_image(
        #         image_name, must_be_grayscale=True
        #     )

        pixels = image.pixel_data

        mask = (ijv[:, 0] < pixels.shape[0]) & (ijv[:, 1] < pixels.shape[1])

        mask[mask] = image.mask[ijv[mask, 0], ijv[mask, 1]]

        yx_ = yx[mask, :]

        l_ = l[mask]

        z_ = z[mask, :]

        if len(l_) == 0:
            for i, (n, m) in enumerate(zernike_indexes):
                ftr = self.get_zernike_magnitude_name( n, m)

                #meas[object_name, ftr] = numpy.zeros(0)
                intensity_distribution_result.append([ftr, np.zeros(0)])

                if self.wants_zernikes == Z_MAGNITUDES_AND_PHASE:
                    ftr = self.get_zernike_phase_name( n, m)

                    #meas[object_name, ftr] = numpy.zeros(0)
                    intensity_distribution_result.append([ftr, np.zeros(0)])


        #    continue

        areas = scipy.ndimage.sum(
            np.ones(l_.shape, int), labels=l_, index=objects.indices
        )

        for i, (n, m) in enumerate(zernike_indexes):
            vr = scipy.ndimage.sum(
                pixels[ijv[mask, 0], ijv[mask, 1]] * z_[:, i].real,
                labels=l_,
                index=objects.indices,
            )

            vi = scipy.ndimage.sum(
                pixels[ijv[mask, 0], ijv[mask, 1]] * z_[:, i].imag,
                labels=l_,
                index=objects.indices,
            )

            magnitude = np.sqrt(vr * vr + vi * vi) / areas

            ftr = self.get_zernike_magnitude_name(n, m)

            #meas[object_name, ftr] = magnitude
            intensity_distribution_result.append([ftr, magnitude])

            if self.wants_zernikes == Z_MAGNITUDES_AND_PHASE:
                phase = np.arctan2(vr, vi)

                ftr = self.get_zernike_phase_name(n, m)

                #meas[object_name, ftr] = phase
                intensity_distribution_result.append([ftr, phase])

        return intensity_distribution_result



    def get_zernike_phase_name(self,  n, m):
        """The feature name of the phase of a Zernike moment

        image_name - the name of the image being measured
        n - the radial moment of the Zernike
        m - the azimuthal moment of the Zernike
        """
        return "_".join((M_CATEGORY, FF_ZERNIKE_PHASE, self.image_name, str(n), str(m)))




    def get_zernike_magnitude_name(self,  n, m):
        """The feature name of the magnitude of a Zernike moment

        image_name - the name of the image being measured
        n - the radial moment of the Zernike
        m - the azimuthal moment of the Zernike
        """
        return "_".join((M_CATEGORY, FF_ZERNIKE_MAGNITUDE, self.image_name, str(n), str(m)))



    def do_measurements(self,image,objects,center_object_name,bin_count_settings,dd):
        """Perform the radial measurements on the image set

         workspace - workspace that holds images / objects
         image_name - make measurements on this image
         object_name - make measurements on these objects
         center_object_name - use the centers of these related objects as
                       the centers for radial measurements. None to use the
                       objects themselves.
         center_choice - the user's center choice for this object:
                       C_SELF, C_CENTERS_OF_OBJECTS or C_EDGES_OF_OBJECTS.
         bin_count_settings - the bin count settings group
         d - a dictionary for saving reusable partial results

         returns one statistics tuple per ring.
         """
        bin_count = bin_count_settings.bin_count

        wants_scaled = bin_count_settings.wants_scaled

        maximum_radius = bin_count_settings.maximum_radius


        labels, pixel_data = crop_labels_and_image(objects.segmented, image.pixel_data)

        nobjects = np.max(objects.segmented)

        measurements_record = []

        heatmaps = {}

        # for heatmap in self.heatmaps:
        #     if (
        #             heatmap.object_name.get_objects_name() == object_name
        #             and image_name == heatmap.image_name.get_image_name()
        #             and heatmap.get_number_of_bins() == bin_count
        #     ):
        #         dd[id(heatmap)] = heatmaps[
        #             MEASUREMENT_ALIASES[heatmap.measurement.value]
        #         ] = numpy.zeros(labels.shape)

        if nobjects == 0:
            for bin_index in range(1, bin_count + 1):
                for feature in (F_FRAC_AT_D, F_MEAN_FRAC, F_RADIAL_CV):
                    feature_name = (feature + FF_GENERIC) % (
                        self.image_name,
                        bin_index,
                        bin_count,
                    )

                    measurements_record.append(["".join([M_CATEGORY, feature_name]),np.zeros(0)])

                    # measurements.add_measurement(
                    #     object_name,
                    #     "_".join([M_CATEGORY, feature_name]),
                    #     numpy.zeros(0),
                    # )

                    if not wants_scaled:
                        measurement_name = "_".join(
                            [M_CATEGORY, feature, self.image_name, FF_OVERFLOW]
                        )

                        measurements_record.append(measurement_name,np.zeros(0))

                        # measurements.add_measurement(
                        #     object_name, measurement_name, numpy.zeros(0)
                        # )
            return measurements_record
            #return [( "no objects", "-", "-", "-", "-")]

        # name = (
        #     object_name
        #     if center_object_name is None
        #     else "{}_{}".format(object_name, center_object_name)
        # )

        # if name in dd:
        #     normalized_distance, i_center, j_center, good_mask = dd[name]
        # else:
        d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)

        # if center_object_name is not None:
        #     #
        #     # Use the center of the centering objects to assign a center
        #     # to each labeled pixel using propagation
        #     #
        #     center_objects = workspace.object_set.get_objects(center_object_name)
        #
        #     center_labels, cmask = size_similarly(labels, center_objects.segmented)
        #
        #     pixel_counts = centrosome.cpmorphology.fixup_scipy_ndimage_result(
        #         scipy.ndimage.sum(
        #             numpy.ones(center_labels.shape),
        #             center_labels,
        #             numpy.arange(
        #                 1, numpy.max(center_labels) + 1, dtype=numpy.int32
        #             ),
        #         )
        #     )
        #
        #     good = pixel_counts > 0
        #
        #     i, j = (
        #             centrosome.cpmorphology.centers_of_labels(center_labels) + 0.5
        #     ).astype(int)
        #
        #     ig = i[good]
        #
        #     jg = j[good]
        #
        #     lg = numpy.arange(1, len(i) + 1)[good]
        #
        #     if center_choice == C_CENTERS_OF_OTHER:
        #         #
        #         # Reduce the propagation labels to the centers of
        #         # the centering objects
        #         #
        #         center_labels = numpy.zeros(center_labels.shape, int)
        #
        #         center_labels[ig, jg] = lg
        #
        #     cl, d_from_center = centrosome.propagate.propagate(
        #         numpy.zeros(center_labels.shape), center_labels, labels != 0, 1
        #     )
        #
        #     #
        #     # Erase the centers that fall outside of labels
        #     #
        #     cl[labels == 0] = 0
        #
        #     #
        #     # If objects are hollow or crescent-shaped, there may be
        #     # objects without center labels. As a backup, find the
        #     # center that is the closest to the center of mass.
        #     #
        #     missing_mask = (labels != 0) & (cl == 0)
        #
        #     missing_labels = numpy.unique(labels[missing_mask])
        #
        #     if len(missing_labels):
        #         all_centers = centrosome.cpmorphology.centers_of_labels(labels)
        #
        #         missing_i_centers, missing_j_centers = all_centers[
        #                                                :, missing_labels - 1
        #                                                ]
        #
        #         di = missing_i_centers[:, numpy.newaxis] - ig[numpy.newaxis, :]
        #
        #         dj = missing_j_centers[:, numpy.newaxis] - jg[numpy.newaxis, :]
        #
        #         missing_best = lg[numpy.argsort(di * di + dj * dj)[:, 0]]
        #
        #         best = numpy.zeros(numpy.max(labels) + 1, int)
        #
        #         best[missing_labels] = missing_best
        #
        #         cl[missing_mask] = best[labels[missing_mask]]
        #
        #         #
        #         # Now compute the crow-flies distance to the centers
        #         # of these pixels from whatever center was assigned to
        #         # the object.
        #         #
        #         iii, jjj = numpy.mgrid[0: labels.shape[0], 0: labels.shape[1]]
        #
        #         di = iii[missing_mask] - i[cl[missing_mask] - 1]
        #
        #         dj = jjj[missing_mask] - j[cl[missing_mask] - 1]
        #
        #         d_from_center[missing_mask] = numpy.sqrt(di * di + dj * dj)
        # else:
        # Find the point in each object farthest away from the edge.
        # This does better than the centroid:
        # * The center is within the object
        # * The center tends to be an interesting point, like the
        #   center of the nucleus or the center of one or the other
        #   of two touching cells.
        #
        i, j = centrosome.cpmorphology.maximum_position_of_labels(
            d_to_edge, labels, objects.indices
        )

        center_labels = np.zeros(labels.shape, int)

        center_labels[i, j] = labels[i, j]

        #
        # Use the coloring trick here to process touching objects
        # in separate operations
        #
        colors = centrosome.cpmorphology.color_labels(labels)

        ncolors = np.max(colors)

        d_from_center = np.zeros(labels.shape)

        cl = np.zeros(labels.shape, int)

        for color in range(1, ncolors + 1):
            mask = colors == color
            l, d = centrosome.propagate.propagate(
                np.zeros(center_labels.shape), center_labels, mask, 1
            )

            d_from_center[mask] = d[mask]

            cl[mask] = l[mask]

        good_mask = cl > 0

        if self.center_choice == C_EDGES_OF_OTHER:
            # Exclude pixels within the centering objects
            # when performing calculations from the centers
            good_mask = good_mask & (center_labels == 0)

        i_center = np.zeros(cl.shape)

        i_center[good_mask] = i[cl[good_mask] - 1]

        j_center = np.zeros(cl.shape)

        j_center[good_mask] = j[cl[good_mask] - 1]

        normalized_distance = np.zeros(labels.shape)

        if wants_scaled:
            total_distance = d_from_center + d_to_edge

            normalized_distance[good_mask] = d_from_center[good_mask] / (
                    total_distance[good_mask] + 0.001
            )
        else:
            normalized_distance[good_mask] = (
                    d_from_center[good_mask] / maximum_radius
            )

        #dd[name] = [normalized_distance, i_center, j_center, good_mask]

        ngood_pixels = np.sum(good_mask)

        good_labels = labels[good_mask]

        bin_indexes = (normalized_distance * bin_count).astype(int)

        bin_indexes[bin_indexes > bin_count] = bin_count

        labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

        histogram = scipy.sparse.coo_matrix(
            (pixel_data[good_mask], labels_and_bins), (nobjects, bin_count + 1)
        ).toarray()

        sum_by_object = np.sum(histogram, 1)

        sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]

        fraction_at_distance = histogram / sum_by_object_per_bin

        number_at_distance = scipy.sparse.coo_matrix(
            (np.ones(ngood_pixels), labels_and_bins), (nobjects, bin_count + 1)
        ).toarray()

        object_mask = number_at_distance > 0

        sum_by_object = np.sum(number_at_distance, 1)

        sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]

        fraction_at_bin = number_at_distance / sum_by_object_per_bin

        mean_pixel_fraction = fraction_at_distance / (
                fraction_at_bin + np.finfo(float).eps
        )

        masked_fraction_at_distance = np.ma.masked_array(
            fraction_at_distance, ~object_mask
        )

        masked_mean_pixel_fraction = np.ma.masked_array(
            mean_pixel_fraction, ~object_mask
        )

        # Anisotropy calculation.  Split each feature into eight wedges, then
        # compute coefficient of variation of the wedges' mean intensities
        # in each ring.
        #
        # Compute each pixel's delta from the center object's centroid
        i, j = np.mgrid[0: labels.shape[0], 0: labels.shape[1]]

        imask = i[good_mask] > i_center[good_mask]

        jmask = j[good_mask] > j_center[good_mask]

        absmask = abs(i[good_mask] - i_center[good_mask]) > abs(
            j[good_mask] - j_center[good_mask]
        )

        radial_index = (
                imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4
        )

        statistics = []

        for bin in range(bin_count + (0 if wants_scaled else 1)):
            bin_mask = good_mask & (bin_indexes == bin)

            bin_pixels = np.sum(bin_mask)

            bin_labels = labels[bin_mask]

            bin_radial_index = radial_index[bin_indexes[good_mask] == bin]

            labels_and_radii = (bin_labels - 1, bin_radial_index)

            radial_values = scipy.sparse.coo_matrix(
                (pixel_data[bin_mask], labels_and_radii), (nobjects, 8)
            ).toarray()

            pixel_count = scipy.sparse.coo_matrix(
                (np.ones(bin_pixels), labels_and_radii), (nobjects, 8)
            ).toarray()

            mask = pixel_count == 0

            radial_means = np.ma.masked_array(radial_values / pixel_count, mask)

            radial_cv = np.std(radial_means, 1) / np.mean(radial_means, 1)

            radial_cv[np.sum(~mask, 1) == 0] = 0

            for measurement, feature, overflow_feature in (
                    (fraction_at_distance[:, bin], MF_FRAC_AT_D, OF_FRAC_AT_D),
                    (mean_pixel_fraction[:, bin], MF_MEAN_FRAC, OF_MEAN_FRAC),
                    (np.array(radial_cv), MF_RADIAL_CV, OF_RADIAL_CV),
            ):
                if bin == bin_count:
                    measurement_name = overflow_feature % self.image_name
                else:
                    measurement_name = feature % (self.image_name, bin + 1, bin_count)

                measurements_record.append([measurement_name, measurement])
                #measurements.add_measurement(object_name, measurement_name, measurement)

                if feature in heatmaps:
                    heatmaps[feature][bin_mask] = measurement[bin_labels - 1]

            radial_cv.mask = np.sum(~mask, 1) == 0

            bin_name = str(bin + 1) if bin < bin_count else "Overflow"

            statistics += [
                (
                    bin_name,
                    str(bin_count),
                    np.round(np.mean(masked_fraction_at_distance[:, bin]), 4),
                    np.round(np.mean(masked_mean_pixel_fraction[:, bin]), 4),
                    np.round(np.mean(radial_cv), 4),
                )
            ]

        return measurements_record








