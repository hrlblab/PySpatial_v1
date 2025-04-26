import numpy as np
from core.objects import Objects
import centrosome.zernike
import centrosome
import skimage as ski
import scipy

AREA_SHAPE = "AreaShape"

F_AREA = "Area"
F_PERIMETER = "Perimeter"
F_VOLUME = "Volume"
F_SURFACE_AREA = "SurfaceArea"
F_ECCENTRICITY = "Eccentricity"
F_SOLIDITY = "Solidity"
F_CONVEX_AREA = "ConvexArea"
F_EXTENT = "Extent"
F_CENTER_X = "Center_X"
F_CENTER_Y = "Center_Y"
F_CENTER_Z = "Center_Z"
F_BBOX_AREA = "BoundingBoxArea"
F_BBOX_VOLUME = "BoundingBoxVolume"
F_MIN_X = "BoundingBoxMinimum_X"
F_MAX_X = "BoundingBoxMaximum_X"
F_MIN_Y = "BoundingBoxMinimum_Y"
F_MAX_Y = "BoundingBoxMaximum_Y"
F_MIN_Z = "BoundingBoxMinimum_Z"
F_MAX_Z = "BoundingBoxMaximum_Z"
F_EULER_NUMBER = "EulerNumber"
F_FORM_FACTOR = "FormFactor"
F_MAJOR_AXIS_LENGTH = "MajorAxisLength"
F_MINOR_AXIS_LENGTH = "MinorAxisLength"
F_ORIENTATION = "Orientation"
F_COMPACTNESS = "Compactness"
F_INERTIA = "InertiaTensor"
F_MAXIMUM_RADIUS = "MaximumRadius"
F_MEDIAN_RADIUS = "MedianRadius"
F_MEAN_RADIUS = "MeanRadius"
F_MIN_FERET_DIAMETER = "MinFeretDiameter"
F_MAX_FERET_DIAMETER = "MaxFeretDiameter"

F_CENTRAL_MOMENT_0_0 = "CentralMoment_0_0"
F_CENTRAL_MOMENT_0_1 = "CentralMoment_0_1"
F_CENTRAL_MOMENT_0_2 = "CentralMoment_0_2"
F_CENTRAL_MOMENT_0_3 = "CentralMoment_0_3"
F_CENTRAL_MOMENT_1_0 = "CentralMoment_1_0"
F_CENTRAL_MOMENT_1_1 = "CentralMoment_1_1"
F_CENTRAL_MOMENT_1_2 = "CentralMoment_1_2"
F_CENTRAL_MOMENT_1_3 = "CentralMoment_1_3"
F_CENTRAL_MOMENT_2_0 = "CentralMoment_2_0"
F_CENTRAL_MOMENT_2_1 = "CentralMoment_2_1"
F_CENTRAL_MOMENT_2_2 = "CentralMoment_2_2"
F_CENTRAL_MOMENT_2_3 = "CentralMoment_2_3"
F_EQUIVALENT_DIAMETER = "EquivalentDiameter"
F_HU_MOMENT_0 = "HuMoment_0"
F_HU_MOMENT_1 = "HuMoment_1"
F_HU_MOMENT_2 = "HuMoment_2"
F_HU_MOMENT_3 = "HuMoment_3"
F_HU_MOMENT_4 = "HuMoment_4"
F_HU_MOMENT_5 = "HuMoment_5"
F_HU_MOMENT_6 = "HuMoment_6"
F_INERTIA_TENSOR_0_0 = "InertiaTensor_0_0"
F_INERTIA_TENSOR_0_1 = "InertiaTensor_0_1"
F_INERTIA_TENSOR_1_0 = "InertiaTensor_1_0"
F_INERTIA_TENSOR_1_1 = "InertiaTensor_1_1"
F_INERTIA_TENSOR_EIGENVALUES_0 = "InertiaTensorEigenvalues_0"
F_INERTIA_TENSOR_EIGENVALUES_1 = "InertiaTensorEigenvalues_1"
F_NORMALIZED_MOMENT_0_0 = "NormalizedMoment_0_0"
F_NORMALIZED_MOMENT_0_1 = "NormalizedMoment_0_1"
F_NORMALIZED_MOMENT_0_2 = "NormalizedMoment_0_2"
F_NORMALIZED_MOMENT_0_3 = "NormalizedMoment_0_3"
F_NORMALIZED_MOMENT_1_0 = "NormalizedMoment_1_0"
F_NORMALIZED_MOMENT_1_1 = "NormalizedMoment_1_1"
F_NORMALIZED_MOMENT_1_2 = "NormalizedMoment_1_2"
F_NORMALIZED_MOMENT_1_3 = "NormalizedMoment_1_3"
F_NORMALIZED_MOMENT_2_0 = "NormalizedMoment_2_0"
F_NORMALIZED_MOMENT_2_1 = "NormalizedMoment_2_1"
F_NORMALIZED_MOMENT_2_2 = "NormalizedMoment_2_2"
F_NORMALIZED_MOMENT_2_3 = "NormalizedMoment_2_3"
F_NORMALIZED_MOMENT_3_0 = "NormalizedMoment_3_0"
F_NORMALIZED_MOMENT_3_1 = "NormalizedMoment_3_1"
F_NORMALIZED_MOMENT_3_2 = "NormalizedMoment_3_2"
F_NORMALIZED_MOMENT_3_3 = "NormalizedMoment_3_3"
F_SPATIAL_MOMENT_0_0 = "SpatialMoment_0_0"
F_SPATIAL_MOMENT_0_1 = "SpatialMoment_0_1"
F_SPATIAL_MOMENT_0_2 = "SpatialMoment_0_2"
F_SPATIAL_MOMENT_0_3 = "SpatialMoment_0_3"
F_SPATIAL_MOMENT_1_0 = "SpatialMoment_1_0"
F_SPATIAL_MOMENT_1_1 = "SpatialMoment_1_1"
F_SPATIAL_MOMENT_1_2 = "SpatialMoment_1_2"
F_SPATIAL_MOMENT_1_3 = "SpatialMoment_1_3"
F_SPATIAL_MOMENT_2_0 = "SpatialMoment_2_0"
F_SPATIAL_MOMENT_2_1 = "SpatialMoment_2_1"
F_SPATIAL_MOMENT_2_2 = "SpatialMoment_2_2"
F_SPATIAL_MOMENT_2_3 = "SpatialMoment_2_3"



ZERNIKE_N=9




class MeasureObjectSizeShape:
    def __init__(self,objects):
        self.calculate_advanced=True
        self.calculate_zernikes=True
        self.objects_list=objects
        self.object_name='IdentifyPrimaryObjects'

    def add_setting(self,calculate_advanced_input,calculate_zernikes_input):
        self.calculate_advanced=calculate_advanced_input
        self.calculate_zernikes =calculate_zernikes_input
        

    def run(self):
        result=[]
        object_id=-1
        for object in self.objects_list:
            object_id=object_id+1
            temp_result=self.run_on_objects(object,object_id)
            if temp_result is not None:
                result+=temp_result

        return result

    def run_on_objects(self, objects,object_id):
        measurement_record=[]

        if len(objects.indices) == 0:
            # There is no object
            return

        if len(objects.shape) ==2:
            #2D
            desired_properties = [
                "label",
                "image",
                "area",
                "perimeter",
                "bbox",
                "bbox_area",
                "major_axis_length",
                "minor_axis_length",
                "orientation",
                "centroid",
                "equivalent_diameter",
                "extent",
                "eccentricity",
                "convex_area",
                "solidity",
                "euler_number",
            ]
            if self.calculate_advanced:
                desired_properties += [
                    "inertia_tensor",
                    "inertia_tensor_eigvals",
                    "moments",
                    "moments_central",
                    "moments_hu",
                    "moments_normalized",
                ]
        else:
           #3D
            desired_properties = [
                "label",
                "image",
                "area",
                "centroid",
                "bbox",
                "bbox_area",
                "major_axis_length",
                "minor_axis_length",
                "extent",
                "equivalent_diameter",
                "euler_number",
            ]
            if self.calculate_advanced.value:
                desired_properties += [
                    "solidity",
                ]

        # check for overlapping object
        if not objects.overlapping():
            features_to_record = self.analyze_objects(objects, desired_properties)
        else:
            # Objects are overlapping, process as single arrays
            coords_array = objects.ijv
            features_to_record = {}
            for label in objects.indices:
                omap = np.zeros(objects.shape)
                ocoords = coords_array[coords_array[:, 2] == label, 0:2]
                np.put(omap, np.ravel_multi_index(ocoords.T, omap.shape), 1)
                tempobject = Objects()
                tempobject.segmented = omap
                buffer = self.analyze_objects(tempobject, desired_properties)
                for f, m in buffer.items():
                    if f in features_to_record:
                        features_to_record[f] = np.concatenate(
                            (features_to_record[f], m)
                        )
                    else:
                        features_to_record[f] = m
                        ##############################

        for f, m in features_to_record.items():
            feature_name=f+"_"+str(object_id)
            measurement_record=self.record_measurement(measurement_record,self.object_name,f,m)
        return measurement_record


    def record_measurement(self, measurement_record, object_name, feature_name, result):
        """Record the result of a measurement in the workspace's measurements"""
        data = centrosome.cpmorphology.fixup_scipy_ndimage_result(result)
        measurement_record.append([ "%s_%s" % (AREA_SHAPE, feature_name), data])
        # workspace.add_measurement(
        #     object_name, "%s_%s" % (AREA_SHAPE, feature_name), data
        # )
        # if self.show_window and np.any(np.isfinite(data)) > 0:
        #     data = data[np.isfinite(data)]
        #     workspace.display_data.statistics.append(
        #         (
        #             object_name,
        #             feature_name,
        #             "%.2f" % np.mean(data),
        #             "%.2f" % np.median(data),
        #             "%.2f" % np.std(data),
        #         )
        #     )
        return measurement_record

    def analyze_objects(self, objects, desired_properties):
        """Computing the measurements for a single map of objects"""
        labels = objects.segmented
        nobjects = len(objects.indices)
        if len(objects.shape) == 2:
            props = ski.measure.regionprops_table(
                labels, properties=desired_properties
            )

            formfactor = 4.0 * np.pi * props["area"] / props["perimeter"] ** 2
            denom = [max(x, 1) for x in 4.0 * np.pi * props["area"]]
            compactness = props["perimeter"] ** 2 / denom

            max_radius = np.zeros(nobjects)
            median_radius = np.zeros(nobjects)
            mean_radius = np.zeros(nobjects)
            min_feret_diameter = np.zeros(nobjects)
            max_feret_diameter = np.zeros(nobjects)
            zernike_numbers = self.get_zernike_numbers()

            zf = {}
            for n, m in zernike_numbers:
                zf[(n, m)] = np.zeros(nobjects)

            for index, mini_image in enumerate(props["image"]):
                # Pad image to assist distance tranform
                mini_image = np.pad(mini_image, 1)
                distances = scipy.ndimage.distance_transform_edt(mini_image)
                max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.maximum(distances, mini_image)
                )
                mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.mean(distances, mini_image)
                )
                median_radius[index] = centrosome.cpmorphology.median_of_labels(
                    distances, mini_image.astype("int"), [1]
                )
            #
            # Zernike features
            #
            if self.calculate_zernikes:
                zf_l = centrosome.zernike.zernike(
                    zernike_numbers, labels, objects.indices
                )
                for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
                    zf[(n, m)] = z

            if nobjects > 0:
                chulls, chull_counts = centrosome.cpmorphology.convex_hull_ijv(
                    objects.ijv, objects.indices
                )
                #
                # Feret diameter
                #
                (
                    min_feret_diameter,
                    max_feret_diameter,
                ) = centrosome.cpmorphology.feret_diameter(
                    chulls, chull_counts, objects.indices
                )

            features_to_record = {
                F_AREA: props["area"],
                F_PERIMETER: props["perimeter"],
                F_MAJOR_AXIS_LENGTH: props["major_axis_length"],
                F_MINOR_AXIS_LENGTH: props["minor_axis_length"],
                F_ECCENTRICITY: props["eccentricity"],
                F_ORIENTATION: props["orientation"] * (180 / np.pi),
                F_CENTER_X: props["centroid-1"],
                F_CENTER_Y: props["centroid-0"],
                F_BBOX_AREA: props["bbox_area"],
                F_MIN_X: props["bbox-1"],
                F_MAX_X: props["bbox-3"],
                F_MIN_Y: props["bbox-0"],
                F_MAX_Y: props["bbox-2"],
                F_FORM_FACTOR: formfactor,
                F_EXTENT: props["extent"],
                F_SOLIDITY: props["solidity"],
                F_COMPACTNESS: compactness,
                F_EULER_NUMBER: props["euler_number"],
                F_MAXIMUM_RADIUS: max_radius,
                F_MEAN_RADIUS: mean_radius,
                F_MEDIAN_RADIUS: median_radius,
                F_CONVEX_AREA: props["convex_area"],
                F_MIN_FERET_DIAMETER: min_feret_diameter,
                F_MAX_FERET_DIAMETER: max_feret_diameter,
                F_EQUIVALENT_DIAMETER: props["equivalent_diameter"],
            }
            if self.calculate_advanced:
                features_to_record.update(
                    {
                        F_SPATIAL_MOMENT_0_0: props["moments-0-0"],
                        F_SPATIAL_MOMENT_0_1: props["moments-0-1"],
                        F_SPATIAL_MOMENT_0_2: props["moments-0-2"],
                        F_SPATIAL_MOMENT_0_3: props["moments-0-3"],
                        F_SPATIAL_MOMENT_1_0: props["moments-1-0"],
                        F_SPATIAL_MOMENT_1_1: props["moments-1-1"],
                        F_SPATIAL_MOMENT_1_2: props["moments-1-2"],
                        F_SPATIAL_MOMENT_1_3: props["moments-1-3"],
                        F_SPATIAL_MOMENT_2_0: props["moments-2-0"],
                        F_SPATIAL_MOMENT_2_1: props["moments-2-1"],
                        F_SPATIAL_MOMENT_2_2: props["moments-2-2"],
                        F_SPATIAL_MOMENT_2_3: props["moments-2-3"],
                        F_CENTRAL_MOMENT_0_0: props["moments_central-0-0"],
                        F_CENTRAL_MOMENT_0_1: props["moments_central-0-1"],
                        F_CENTRAL_MOMENT_0_2: props["moments_central-0-2"],
                        F_CENTRAL_MOMENT_0_3: props["moments_central-0-3"],
                        F_CENTRAL_MOMENT_1_0: props["moments_central-1-0"],
                        F_CENTRAL_MOMENT_1_1: props["moments_central-1-1"],
                        F_CENTRAL_MOMENT_1_2: props["moments_central-1-2"],
                        F_CENTRAL_MOMENT_1_3: props["moments_central-1-3"],
                        F_CENTRAL_MOMENT_2_0: props["moments_central-2-0"],
                        F_CENTRAL_MOMENT_2_1: props["moments_central-2-1"],
                        F_CENTRAL_MOMENT_2_2: props["moments_central-2-2"],
                        F_CENTRAL_MOMENT_2_3: props["moments_central-2-3"],
                        F_NORMALIZED_MOMENT_0_0: props["moments_normalized-0-0"],
                        F_NORMALIZED_MOMENT_0_1: props["moments_normalized-0-1"],
                        F_NORMALIZED_MOMENT_0_2: props["moments_normalized-0-2"],
                        F_NORMALIZED_MOMENT_0_3: props["moments_normalized-0-3"],
                        F_NORMALIZED_MOMENT_1_0: props["moments_normalized-1-0"],
                        F_NORMALIZED_MOMENT_1_1: props["moments_normalized-1-1"],
                        F_NORMALIZED_MOMENT_1_2: props["moments_normalized-1-2"],
                        F_NORMALIZED_MOMENT_1_3: props["moments_normalized-1-3"],
                        F_NORMALIZED_MOMENT_2_0: props["moments_normalized-2-0"],
                        F_NORMALIZED_MOMENT_2_1: props["moments_normalized-2-1"],
                        F_NORMALIZED_MOMENT_2_2: props["moments_normalized-2-2"],
                        F_NORMALIZED_MOMENT_2_3: props["moments_normalized-2-3"],
                        F_NORMALIZED_MOMENT_3_0: props["moments_normalized-3-0"],
                        F_NORMALIZED_MOMENT_3_1: props["moments_normalized-3-1"],
                        F_NORMALIZED_MOMENT_3_2: props["moments_normalized-3-2"],
                        F_NORMALIZED_MOMENT_3_3: props["moments_normalized-3-3"],
                        F_HU_MOMENT_0: props["moments_hu-0"],
                        F_HU_MOMENT_1: props["moments_hu-1"],
                        F_HU_MOMENT_2: props["moments_hu-2"],
                        F_HU_MOMENT_3: props["moments_hu-3"],
                        F_HU_MOMENT_4: props["moments_hu-4"],
                        F_HU_MOMENT_5: props["moments_hu-5"],
                        F_HU_MOMENT_6: props["moments_hu-6"],
                        F_INERTIA_TENSOR_0_0: props["inertia_tensor-0-0"],
                        F_INERTIA_TENSOR_0_1: props["inertia_tensor-0-1"],
                        F_INERTIA_TENSOR_1_0: props["inertia_tensor-1-0"],
                        F_INERTIA_TENSOR_1_1: props["inertia_tensor-1-1"],
                        F_INERTIA_TENSOR_EIGENVALUES_0: props[
                            "inertia_tensor_eigvals-0"
                        ],
                        F_INERTIA_TENSOR_EIGENVALUES_1: props[
                            "inertia_tensor_eigvals-1"
                        ],
                    }
                )

            if self.calculate_zernikes:
                features_to_record.update(
                    {
                        self.get_zernike_name((n, m)): zf[(n, m)]
                        for n, m in zernike_numbers
                    }
                )

        else:

            props = ski.measure.regionprops_table(
                labels, properties=desired_properties
            )

            # SurfaceArea
            surface_areas = np.zeros(len(props["label"]))
            for index, label in enumerate(props["label"]):
                # this seems less elegant than you might wish, given that regionprops returns a slice,
                # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
                volume= labels[max(props['bbox-0'][index]-1,0):min(props['bbox-3'][index]+1,labels.shape[0]),
                          max(props['bbox-1'][index]-1,0):min(props['bbox-4'][index]+1,labels.shape[1]),
                          max(props['bbox-2'][index]-1,0):min(props['bbox-5'][index]+1,labels.shape[2])]
                volume = volume == label
                verts, faces, _normals, _values = ski.measure.marching_cubes(
                    volume,
                    method="lewiner",
                    spacing=objects.parent_image.spacing
                    if objects.has_parent_image
                    else (1.0,) * labels.ndim,
                    level=0,
                )
                surface_areas[index] = ski.measure.mesh_surface_area(verts, faces)

            features_to_record = {
                F_VOLUME: props["area"],
                F_SURFACE_AREA: surface_areas,
                F_MAJOR_AXIS_LENGTH: props["major_axis_length"],
                F_MINOR_AXIS_LENGTH: props["minor_axis_length"],
                F_CENTER_X: props["centroid-2"],
                F_CENTER_Y: props["centroid-1"],
                F_CENTER_Z: props["centroid-0"],
                F_BBOX_VOLUME: props["bbox_area"],
                F_MIN_X: props["bbox-2"],
                F_MAX_X: props["bbox-5"],
                F_MIN_Y: props["bbox-1"],
                F_MAX_Y: props["bbox-4"],
                F_MIN_Z: props["bbox-0"],
                F_MAX_Z: props["bbox-3"],
                F_EXTENT: props["extent"],
                F_EULER_NUMBER: props["euler_number"],
                F_EQUIVALENT_DIAMETER: props["equivalent_diameter"],
            }
            if self.calculate_advanced.value:
                features_to_record[F_SOLIDITY] = props["solidity"]
        return features_to_record

    def get_zernike_name(self, zernike_index):
        """Return the name of a Zernike feature, given a (N,M) 2-tuple

        zernike_index - a 2 element sequence organized as N,M
        """
        return "Zernike_%d_%d" % (zernike_index[0], zernike_index[1])


    def get_zernike_numbers(self):
        """The Zernike numbers measured by this module"""
        if self.calculate_zernikes:
            return centrosome.zernike.get_zernike_indexes(ZERNIKE_N + 1)
        else:
            return []


