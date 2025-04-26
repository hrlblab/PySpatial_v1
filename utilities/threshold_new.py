import numpy as np
import centrosome
from utilities.image_processing import ImageProcessing
from skimage import filters
import skimage as ski
import scipy

O_TWO_CLASS = "Two classes"
O_THREE_CLASS = "Three classes"

O_FOREGROUND = "Foreground"
O_BACKGROUND = "Background"

RB_MEAN = "Mean"
RB_MEDIAN = "Median"
RB_MODE = "Mode"
RB_SD = "Standard deviation"
RB_MAD = "Median absolute deviation"

TS_GLOBAL = "Global"
TS_ADAPTIVE = "Adaptive"
TM_MANUAL = "Manual"
TM_MEASUREMENT = "Measurement"
TM_LI = "Minimum Cross-Entropy"
TM_OTSU = "Otsu"
TM_ROBUST_BACKGROUND = "Robust Background"
TM_SAUVOLA = "Sauvola"


class Threshold():
    threshold_operation='Otsu'
    log_transform=False
    threshold_scope='Adaptive'
    two_class_otsu = "Two classes"
    assign_middle_to_foreground="Foreground"
    adaptive_window_size=50
    threshold_correction_factor=1.0
    threshold_range=(0,1)
    threshold_smoothing_scale=0.5

    def add_setting(self,threshold_scope_input,global_operation_input,local_operation_input,threshold_smoothing_scale_input,
                    threshold_correction_factor_input,threshold_range_input,manual_threshold_input,thresholding_measurement_input,
                    two_class_otsu_input,assign_middle_to_foreground_input,lower_outlier_fraction_input,upper_outlier_fraction_input,
                    averaging_method_input,variance_method_input,number_of_deviations_input,adaptive_window_size_input,log_transform_input,threshold_operation_input):

        self.threshold_scope = threshold_scope_input

        self.global_operation = global_operation_input

        self.local_operation = local_operation_input

        self.threshold_smoothing_scale = threshold_smoothing_scale_input

        self.threshold_correction_factor = threshold_correction_factor_input

        self.threshold_range = threshold_range_input

        self.manual_threshold = manual_threshold_input

        self.thresholding_measurement = thresholding_measurement_input

        self.two_class_otsu = two_class_otsu_input


        self.assign_middle_to_foreground = assign_middle_to_foreground_input

        self.lower_outlier_fraction = lower_outlier_fraction_input

        self.upper_outlier_fraction = upper_outlier_fraction_input

        self.averaging_method = averaging_method_input


        self.variance_method = variance_method_input


        self.number_of_deviations = number_of_deviations_input


        self.adaptive_window_size = adaptive_window_size_input

        self.log_transform = log_transform_input
        #self.threshold_operation="Minimum Cross-Entropy"
        #self.threshold_operation = "Otsu"
        self.threshold_operation =threshold_operation_input




    def _init_(self):
        return 0

      # library 中的threshold
    def library_threshold(self,
        image,
        mask=None,
        threshold_scope="global",
        threshold_method="otsu",
        assign_middle_to_foreground="foreground",
        log_transform=False,
        threshold_correction_factor=1,
        threshold_min=0,
        threshold_max=1,
        window_size=50,
        smoothing=0,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method="mean",
        variance_method="standard_deviation",
        number_of_deviations=2,
        volumetric=False,
        automatic=False,
        **kwargs,
        ):

        # if threshold_scope== adaptive
        final_threshold = ImageProcessing.get_adaptive_threshold(
                  image=image,
                  mask=mask,
                  threshold_method=threshold_method,
                  window_size=window_size,
                  threshold_min=threshold_min,
                  threshold_max=threshold_max,
                  threshold_correction_factor=threshold_correction_factor,
                  assign_middle_to_foreground=assign_middle_to_foreground,
                  log_transform=log_transform,
                  volumetric=volumetric,
                  **kwargs,
        )

        orig_threshold = ImageProcessing.get_adaptive_threshold(
                  image,
                  mask=mask,
                  threshold_method=threshold_method,
                  window_size=window_size,
                  # If automatic=True, do not correct the threshold
                  threshold_min=threshold_min if automatic else 0,
                  threshold_max=threshold_max if automatic else 1,
                  threshold_correction_factor=threshold_correction_factor if automatic else 1,
                  assign_middle_to_foreground=assign_middle_to_foreground,
                  log_transform=log_transform,
                  volumetric=volumetric,
                  **kwargs,
        )

        guide_threshold = ImageProcessing.get_global_threshold(
                  image,
                  mask=mask,
                  threshold_method=threshold_method,
                  threshold_min=threshold_min,
                  threshold_max=threshold_max,
                  threshold_correction_factor=threshold_correction_factor,
                  assign_middle_to_foreground=assign_middle_to_foreground,
                  log_transform=log_transform,
                  **kwargs,
        )

        binary_image, sigma =ImageProcessing.apply_threshold(
                  image,
                  threshold=final_threshold,
                  mask=mask,
                  smoothing=smoothing,
        )

        ##end if
        return final_threshold, orig_threshold, guide_threshold, binary_image, sigma



    def get_threshold(self, image,mask, volumetric, automatic=False):
        need_transform = (
                not automatic and
                self.threshold_operation in (TM_LI, TM_OTSU) and
                self.log_transform
        )

        if need_transform:
            image_data, conversion_dict = centrosome.threshold.log_transform(image)
        else:
            image_data = image

        if self.threshold_operation == TM_MANUAL:
            return self.manual_threshold.value, self.manual_threshold.value, None

    #    elif self.threshold_operation == TM_MEASUREMENT:
            #         # Thresholds are stored as single element arrays.  Cast to float to extract the value.
            # orig_threshold = float(
            #     workspace.measurements.get_current_image_measurement(
            #         self.thresholding_measurement.value
            #     )
            # )
            # return self._correct_global_threshold(orig_threshold), orig_threshold, None

        elif self.threshold_scope == TS_GLOBAL or automatic:
            th_guide = None
            th_original = self.get_global_threshold(image_data, mask, automatic=automatic)

        elif self.threshold_scope == TS_ADAPTIVE:
            th_guide = self.get_global_threshold(image_data, mask)
            th_original = self.get_local_threshold(image_data, mask, volumetric)
        else:
            raise ValueError("Invalid thresholding settings")

        if need_transform:
            th_original = centrosome.threshold.inverse_log_transform(th_original, conversion_dict)
            if th_guide is not None:
                th_guide = centrosome.threshold.inverse_log_transform(th_guide, conversion_dict)

        if self.threshold_scope == TS_GLOBAL or automatic:
            th_corrected = self._correct_global_threshold(th_original)
        else:
            th_guide = self._correct_global_threshold(th_guide)
            th_corrected = self._correct_local_threshold(th_original, th_guide)

        return th_corrected, th_original, th_guide


    def get_global_threshold(self, image, mask, automatic=False):
        image_data = image[mask]

        # Shortcuts - Check if image array is empty or all pixels are the same value.
        if len(image_data) == 0:
            threshold = 0.0

        elif np.all(image_data == image_data[0]):
            threshold = image_data[0]

        elif automatic or self.threshold_operation in (TM_LI, TM_SAUVOLA):
            tol = max(np.min(np.diff(np.unique(image_data))) / 2, 0.5 / 65536)
            threshold = filters.threshold_li(image_data, tolerance=tol)

        elif self.threshold_operation == TM_ROBUST_BACKGROUND:
            threshold = self.get_threshold_robust_background(image_data)

        elif self.threshold_operation == TM_OTSU:
            if self.two_class_otsu == O_TWO_CLASS:
                threshold = filters.threshold_otsu(image_data)
            elif self.two_class_otsu == O_THREE_CLASS:
                bin_wanted = (
                    0 if self.assign_middle_to_foreground.value == "Foreground" else 1
                )
                threshold = filters.threshold_multiotsu(image_data, nbins=128)
                threshold = threshold[bin_wanted]
        else:
            raise ValueError("Invalid thresholding settings")
        return threshold


    def get_local_threshold(self, image, mask, volumetric):
        image_data = np.where(mask, image, np.nan)

        if len(image_data) == 0 or np.all(image_data == np.nan):
            local_threshold = np.zeros_like(image_data)

        elif np.all(image_data == image_data[0]):
            local_threshold = np.full_like(image_data, image_data[0])

        elif self.threshold_operation == TM_LI:
            local_threshold = self._run_local_threshold(
                image_data,
                method=filters.threshold_li,
                volumetric=volumetric,
                tolerance=max(np.min(np.diff(np.unique(image))) / 2, 0.5 / 65536)
            )
        elif self.threshold_operation == TM_OTSU:
            if self.two_class_otsu == O_TWO_CLASS:
                local_threshold = self._run_local_threshold(
                    image_data,
                    method=filters.threshold_otsu,
                    volumetric=volumetric,
                )

            elif self.two_class_otsu == O_THREE_CLASS:
                local_threshold = self._run_local_threshold(
                    image_data,
                    method=filters.threshold_multiotsu,
                    volumetric=volumetric,
                    nbins=128,
                )

        elif self.threshold_operation == TM_ROBUST_BACKGROUND:
            local_threshold = self._run_local_threshold(
                image_data,
                method=self.get_threshold_robust_background,
                volumetric=volumetric,
            )

        elif self.threshold_operation == TM_SAUVOLA:
            image_data = np.where(mask, image, 0)
            adaptive_window = self.adaptive_window_size.value
            if adaptive_window % 2 == 0:
                adaptive_window += 1
            local_threshold = filters.threshold_sauvola(
                image_data, window_size=adaptive_window
            )

        else:
            raise ValueError("Invalid thresholding settings")
        return local_threshold

    def add_image(self,input_image):
        self.image=input_image

    def _run_local_threshold(self, image_data, method, volumetric=False, **kwargs):
        if volumetric:
            t_local = np.zeros_like(image_data)
            for index, plane in enumerate(image_data):
                t_local[index] = self._get_adaptive_threshold(plane, method, **kwargs)
        else:
            t_local = self._get_adaptive_threshold(image_data, method, **kwargs)
        return ski.img_as_float(t_local)


    def _correct_global_threshold(self, threshold):
        threshold *= self.threshold_correction_factor
        return min(max(threshold, min(self.threshold_range)), max(self.threshold_range))


    def apply_threshold(self, image,input_mask, threshold, automatic=False):
        data = image

        mask = input_mask

        if not automatic and self.threshold_smoothing_scale == 0:
            return (data >= threshold) & mask, 0

        if automatic:
            sigma = 1
        else:
            # Convert from a scale into a sigma. What I've done here
            # is to structure the Gaussian so that 1/2 of the smoothed
            # intensity is contributed from within the smoothing diameter
            # and 1/2 is contributed from outside.
            sigma = self.threshold_smoothing_scale / 0.6744 / 2.0

        blurred_image = centrosome.smooth.smooth_with_function_and_mask(
            data,
            lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
            mask,
        )

        return (blurred_image >= threshold) & mask, sigma

    def run(self, workspace):
        input_image = self.image
        dimensions = input_image.dimensions
        final_threshold, orig_threshold, guide_threshold = self.get_threshold(
            input_image, workspace, automatic=False,
        )

        self.add_threshold_measurements(
            self.get_measurement_objects_name(),
            workspace.measurements,
            final_threshold,
            orig_threshold,
            guide_threshold,
        )

        binary_image, _ = self.apply_threshold(input_image, final_threshold)

        self.add_fg_bg_measurements(
            self.get_measurement_objects_name(),
            workspace.measurements,
            input_image,
            binary_image,
        )

        output = Image(binary_image, parent_image=input_image, dimensions=dimensions)

        workspace.image_set.add(self.y_name.value, output)

        if self.show_window:
            workspace.display_data.input_pixel_data = input_image.pixel_data
            workspace.display_data.output_pixel_data = output.pixel_data
            workspace.display_data.dimensions = dimensions
            statistics = workspace.display_data.statistics = []
            workspace.display_data.col_labels = ("Feature", "Value")
            if self.threshold_scope == TS_ADAPTIVE:
                workspace.display_data.threshold_image = final_threshold

            for column in self.get_measurement_columns(workspace.pipeline):
                value = workspace.measurements.get_current_image_measurement(column[1])
                statistics += [(column[1].split("_")[1], str(value))]


    def _get_adaptive_threshold(self, image_data, threshold_method, **kwargs):
        """Given a global threshold, compute a threshold per pixel

        Break the image into blocks, computing the threshold per block.
        Afterwards, constrain the block threshold to .7 T < t < 1.5 T.
        """
        # for the X and Y direction, find the # of blocks, given the
        # size constraints
        if self.threshold_operation == TM_OTSU:
            bin_wanted = (
                0 if self.assign_middle_to_foreground == "Foreground" else 1
            )
        image_size = np.array(image_data.shape[:2], dtype=int)
        nblocks = image_size // self.adaptive_window_size
        if any(n < 2 for n in nblocks):
            raise ValueError(
                "Adaptive window cannot exceed 50%% of an image dimension.\n"
                "Window of %dpx is too large for a %sx%s image"
                % (self.adaptive_window_size, image_size[1], image_size[0])
            )
        #
        # Use a floating point block size to apportion the roundoff
        # roughly equally to each block
        #
        increment = np.array(image_size, dtype=float) / np.array(
            nblocks, dtype=float
        )
        #
        # Put the answer here
        #
        thresh_out = np.zeros(image_size, image_data.dtype)
        #
        # Loop once per block, computing the "global" threshold within the
        # block.
        #
        block_threshold = np.zeros([nblocks[0], nblocks[1]])
        for i in range(nblocks[0]):
            i0 = int(i * increment[0])
            i1 = int((i + 1) * increment[0])
            for j in range(nblocks[1]):
                j0 = int(j * increment[1])
                j1 = int((j + 1) * increment[1])
                block = image_data[i0:i1, j0:j1]
                block = block[~np.isnan(block)]
                if len(block) == 0:
                    threshold_out = 0.0
                elif np.all(block == block[0]):
                    # Don't compute blocks with only 1 value.
                    threshold_out = block[0]
                elif (self.threshold_operation == TM_OTSU and
                      self.two_class_otsu == O_THREE_CLASS and
                      len(np.unique(block)) < 3):
                    # Can't run 3-class otsu on only 2 values.
                    threshold_out = filters.threshold_otsu(block)
                else:
                    try:
                        threshold_out = threshold_method(block, **kwargs)
                    except ValueError:
                        # Drop nbins kwarg when multi-otsu fails. See issue #6324 scikit-image
                        threshold_out = threshold_method(block)
                if isinstance(threshold_out, np.ndarray):
                    # Select correct bin if running multiotsu
                    threshold_out = threshold_out[bin_wanted]
                block_threshold[i, j] = threshold_out

        #
        # Use a cubic spline to blend the thresholds across the image to avoid image artifacts
        #
        spline_order = min(3, np.min(nblocks) - 1)
        xStart = int(increment[0] / 2)
        xEnd = int((nblocks[0] - 0.5) * increment[0])
        yStart = int(increment[1] / 2)
        yEnd = int((nblocks[1] - 0.5) * increment[1])
        xtStart = 0.5
        xtEnd = image_data.shape[0] - 0.5
        ytStart = 0.5
        ytEnd = image_data.shape[1] - 0.5
        block_x_coords = np.linspace(xStart, xEnd, nblocks[0])
        block_y_coords = np.linspace(yStart, yEnd, nblocks[1])
        adaptive_interpolation = scipy.interpolate.RectBivariateSpline(
            block_x_coords,
            block_y_coords,
            block_threshold,
            bbox=(xtStart, xtEnd, ytStart, ytEnd),
            kx=spline_order,
            ky=spline_order,
        )
        thresh_out_x_coords = np.linspace(
            0.5, int(nblocks[0] * increment[0]) - 0.5, thresh_out.shape[0]
        )
        thresh_out_y_coords = np.linspace(
            0.5, int(nblocks[1] * increment[1]) - 0.5, thresh_out.shape[1]
        )

        thresh_out = adaptive_interpolation(thresh_out_x_coords, thresh_out_y_coords)

        return thresh_out


    def add_threshold_measurements(self,final_threshold,orig_threshold,guide_threshold=None,):
        ave_final_threshold = np.mean(np.atleast_1d(final_threshold))
        ave_orig_threshold = np.mean(np.atleast_1d(orig_threshold))
        #if threshold_scope =="Adaptive":
        return ave_final_threshold,ave_orig_threshold,guide_threshold


    def add_fg_bg_measurements( self,image,input_mask, binary_image):
        #data = image.pixel_data
        data = image
        mask = input_mask
        wv = centrosome.threshold.weighted_variance(data, mask, binary_image)
        FF_WEIGHTED_VARIANCE=np.array([wv], dtype=float)

        entropies = centrosome.threshold.sum_of_entropies(data, mask, binary_image)
        FF_SUM_OF_ENTROPIES=np.array([entropies], dtype=float)

        return FF_WEIGHTED_VARIANCE,FF_SUM_OF_ENTROPIES


    def _correct_local_threshold(self, t_local_orig, t_guide):
        t_local = t_local_orig.copy()
        t_local *= self.threshold_correction_factor

        # Constrain the local threshold to be within [0.7, 1.5] * global_threshold. It's for the pretty common case
        # where you have regions of the image with no cells whatsoever that are as large as whatever window you're
        # using. Without a lower bound, you start having crazy threshold s that detect noise blobs. And same for
        # very crowded areas where there is zero background in the window. You want the foreground to be all
        # detected.
        t_min = max(min(self.threshold_range), t_guide * 0.7)
        t_max = min(max(self.threshold_range), t_guide * 1.5)

        t_local[t_local < t_min] = t_min
        t_local[t_local > t_max] = t_max

        return t_local




