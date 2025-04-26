import numpy
import skimage
import mahotas.features

from utilities.object import size_similarly
from setting._hidden_count import HiddenCount
from setting._settings_group import SettingsGroup


IO_IMAGES = "Images"
IO_OBJECTS = "Objects"
IO_BOTH = "Both"
TEXTURE = "Texture"

F_HARALICK = """AngularSecondMoment Contrast Correlation Variance
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()


class MeasureTexture:
    def __init__(self,volumetric_input):
        self.images_or_objects="Both"
        self.gray_levels=256 #enter how many gray levels to measure tje texture at
        #self.scale=2 # texture scale to measure
        self.volumetric=False
        self.volumetric=volumetric_input
        self.image_name="podocyte_nuclei"
        self.object_name="IdentifyPrimaryObjects"
        print("MeasureTexture")

    def add_setting(self,images_or_objects_input,gray_levels_input,scale_input,image_list_input):
        self.images_or_objects=images_or_objects_input
        self.gray_levels=gray_levels_input
        self.images_list = image_list_input

        self.scale_groups = []
        self.scale_count = HiddenCount(self.scale_groups)
        self.add_scale(scale_input)


    def add_scale(self,scale_input):
        group = SettingsGroup()
        group.append("scale", scale_input)
        self.scale_groups.append(group)



    def add_object(self, object_list_input):
        if self.images_or_objects==IO_IMAGES:
            print("Dont't need to add object")
        else:
            self.objects_list=object_list_input




    def run(self):
        result=[]
        image_id=-1
        for image in self.images_list:
            for scale_group in self.scale_groups:
                scale = scale_group.scale
                image_id=image_id+1

                if self.wants_image_measurements():
                    temp_image_result=self.run_image(image, scale,self.image_name)
                    if temp_image_result is not None:
                        result += temp_image_result

                if self.wants_object_measurements():
                    object_id=-1
                    for object in self.objects_list:
                        object_id=object_id+1
                        temp_object_result=self.run_one(
                            image, object, scale,self.image_name,self.object_name
                        )
                        if temp_object_result is not None:
                            result += temp_object_result
        return result


    def wants_image_measurements(self):
        return self.images_or_objects in (IO_IMAGES, IO_BOTH)


    def wants_object_measurements(self):
        return self.images_or_objects in (IO_OBJECTS, IO_BOTH)



    def run_image(self, image, scale,image_id):
        record_measurement = []

        # mahotas.features.haralick bricks itself when provided a dtype larger than uint8 (version 1.4.3)
        gray_levels = int(self.gray_levels)
        pixel_data = skimage.util.img_as_ubyte(image.pixel_data)
        if gray_levels != 256:
            pixel_data = skimage.exposure.rescale_intensity(
                pixel_data, in_range=(0, 255), out_range=(0, gray_levels - 1)
            ).astype(numpy.uint8)

        features = mahotas.features.haralick(pixel_data, distance=scale)

        for direction, direction_features in enumerate(features):
            object_name = "{:d}_{:02d}".format(scale, direction)

            for feature_name, feature in zip(F_HARALICK, direction_features):
                record_measurement += self.record_image_measurement(
                    feature_name=feature_name,
                    image_id=image_id,
                    result=feature,
                    scale=object_name,
                    gray_levels="{:d}".format(gray_levels),
                )

        return record_measurement



    def record_image_measurement(
        self,  image_id, scale, feature_name, result, gray_levels
    ):
        # TODO: this is very concerning
        if not numpy.isfinite(result):
            result = 0

        feature = "{}_{}_{}_{}_{}".format(
            TEXTURE, feature_name, image_id, str(scale), gray_levels
        )

        self_record_measurement=[]
        self_record_measurement.append([feature, result])

        return self_record_measurement


    def run_one(self, image, objects, scale,image_id,object_id):
        measurement_record = []

        labels = objects.segmented

        gray_levels = int(self.gray_levels)

        unique_labels = numpy.unique(labels)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        n_directions = 13 if objects.volumetric else 4

        if len(unique_labels) == 0:
            for direction in range(n_directions):
                for feature_name in F_HARALICK:
                    measurement_record += self.record_measurement(
                        image=self.image_name,
                        feature=feature_name,
                        obj=self.object_name,
                        result=numpy.zeros((0,)),
                        scale="{:d}_{:02d}".format(scale, direction),
                        gray_levels="{:d}".format(gray_levels),
                    )

            return measurement_record

        # IMG-961: Ensure image and objects have the same shape.
        try:
            mask = (
                image.mask
                if image.has_mask
                else numpy.ones_like(image.pixel_data, dtype=bool)
            )
            pixel_data = objects.crop_image_similarly(image.pixel_data)
        except ValueError:
            pixel_data, m1 = size_similarly(labels, image.pixel_data)

            if numpy.any(~m1):
                if image.has_mask:
                    mask, m2 = size_similarly(labels, image.mask)
                    mask[~m2] = False
                else:
                    mask = m1

        pixel_data[~mask] = 0
        # mahotas.features.haralick bricks itself when provided a dtype larger than uint8 (version 1.4.3)
        pixel_data = skimage.util.img_as_ubyte(pixel_data)
        if gray_levels != 256:
            pixel_data = skimage.exposure.rescale_intensity(
                pixel_data, in_range=(0, 255), out_range=(0, gray_levels - 1)
            ).astype(numpy.uint8)
        props = skimage.measure.regionprops(labels, pixel_data)

        features = numpy.empty((n_directions, 13, len(unique_labels)))

        for index, prop in enumerate(props):
            label_data = prop["intensity_image"]
            try:
                features[:, :, index] = mahotas.features.haralick(
                    label_data, distance=scale, ignore_zeros=True
                )
            except ValueError:
                features[:, :, index] = numpy.nan

        for direction, direction_features in enumerate(features):
            for feature_name, feature in zip(F_HARALICK, direction_features):
                measurement_record += self.record_measurement(
                    image=image_id,
                    feature=feature_name,
                    obj=object_id,
                    result=feature,
                    scale="{:d}_{:02d}".format(scale, direction),
                    gray_levels="{:d}".format(gray_levels),
                )

        return measurement_record

    def record_measurement(
            self,  image, obj, scale, feature, result, gray_levels
    ):
        result[~numpy.isfinite(result)] = 0

        self_record_measurement=[]
        self_record_measurement.append(["{}_{}_{}_{}_{}".format(TEXTURE, feature, image, str(scale), gray_levels),
            result,])


        return self_record_measurement









