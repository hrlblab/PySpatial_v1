import os
# enable cv2 to read large file
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
from PIL import Image,ImageDraw
import PIL.Image
import json
from shapely.geometry import Polygon, Point,MultiPolygon
import math

from core.patch import Patch
from core.image_merged import Image_Pack
from feature.measure_object_size_shape import MeasureObjectSizeShape
from feature.measure_texture import  MeasureTexture
from feature.measure_object_intensity import MeasureObjectIntensity
from feature.measure_object_intensity_distribution import MeasureObjectIntensityDistribution
import numpy as np
from feature.identify_primary_object_new import IdentifyPrimaryObject
from scipy.sparse import csr_matrix
import pandas as pd
from feature.get_connect_component_object import GetConnectComponentObject

result_base_path="example/result/PEC_result"
Image.MAX_IMAGE_PIXELS = 99999999999


wsi_path="example/data/PEC/13-269_series0.tiff"


# read the image

whole_image = PIL.Image.open(wsi_path)


# read the geojson file
geofile_path="example/data/PEC/13-269.czi.geojson"

image_pack_instance=Image_Pack()

##
padding_size=3


with open(geofile_path, 'r') as geojson_file:
    geojson_data = json.load(geojson_file)
    feature_list = geojson_data['features']

    polygons = []

    for feature in feature_list:
        coordinates = feature['geometry']['coordinates']
        geom_type = feature['geometry']['type']

        if geom_type == 'Polygon':
            polygon_coords = [(float(coord[0]), float(coord[1])) for coord in coordinates[0]]
            polygons.append(Polygon(polygon_coords))

        elif geom_type == 'MultiPolygon':
            multi_polygons = []
            for part in coordinates:
                polygon_coords = [(float(coord[0]), float(coord[1])) for coord in part[0]]
                polygons.append(Polygon(polygon_coords))
            polygons.append(MultiPolygon(multi_polygons))


    i = 0
    for poly in polygons:
        # get bounding box (minx, miny, maxx, maxy)
        bbox = poly.bounds
        # print(i)

        if not math.isnan(bbox[0]):
            #make new patch
            new_img = PIL.Image.new('L', (int(bbox[2])-int(bbox[0])+padding_size*2, int(bbox[3])-int(bbox[1])+padding_size*2), color='black')
            new_draw = ImageDraw.Draw(new_img)

            # get the sparse matrix
            #from bbx-1
            column=[x-math.floor(bbox[0])+padding_size for y in range(math.floor(bbox[1]), math.ceil(bbox[3])) for x in range(math.floor(bbox[0]), math.ceil(bbox[2])) if poly.contains((Point(x, y)))]
            row = [y-math.floor(bbox[1])+padding_size for y in range(math.floor(bbox[1]), math.ceil(bbox[3])) for x in
                      range(math.floor(bbox[0]), math.ceil(bbox[2])) if poly.contains((Point(x, y)))]
            data=np.ones_like(row)
            sparse_matrix = csr_matrix((data, (row, column)), shape=(math.ceil(bbox[3])-math.floor(bbox[1])+padding_size*2, math.ceil(bbox[2])-math.floor(bbox[0])+padding_size*2))
            new_mask=sparse_matrix.toarray()

            image_pack_instance.addpatch(Patch((np.array(whole_image.crop((math.floor(bbox[0]) - padding_size,math.floor(bbox[1]) - padding_size,math.ceil(bbox[2]) + padding_size,math.ceil(bbox[3]) + padding_size)).convert(mode="L"),dtype=np.float64)/255)*new_mask.astype(np.float64),new_mask.astype(np.uint8), i,[math.floor(bbox[0]) - padding_size,math.floor(bbox[1]) - padding_size,math.ceil(bbox[2]) + padding_size,math.ceil(bbox[3]) + padding_size]))
            i = i + 1




#merge patch

image_pack_instance.pack_rectangles(2000,2000)

# calculate the feature
# add to image class
from core.image import  Image
image_instance = Image(
    image_pack_instance.merged_image[0],
    mask=image_pack_instance.merged_mask[0].astype('bool'),
    path_name=None,
    file_name=None,
    scale=None,
    channelstack=image_pack_instance.merged_image[0].ndim == 3 and image_pack_instance.merged_image[0].shape[-1] > 3,
    #uint8_image=None,
    convert=False
)
# calculate the feature





# identify primary object
get_connect_ins=GetConnectComponentObject()
objects=get_connect_ins.run(image_pack_instance.merged_mask[0].astype(np.uint8),image_instance)

image_instance.masking_objects=objects



temp_obj=objects.segmented.copy()
temp_obj[temp_obj>=1]=1





# measure object size shape
measure_size_shape_ins = MeasureObjectSizeShape([objects])
measure_size_shape_ins.add_setting(calculate_advanced_input=True, calculate_zernikes_input=True)
shape_feature_record = measure_size_shape_ins.run()


# measure object texture
measure_texture_ins = MeasureTexture(False)
measure_texture_ins.add_setting("Objects", 256, 2, [image_instance])
measure_texture_ins.add_object([objects])
measure_texture_record = measure_texture_ins.run()



# measure object intensity
measure_object_intensity_ins = MeasureObjectIntensity([image_instance], [objects])
measure_object_intensity_record = measure_object_intensity_ins.run()



measure_object_intensity_distribution_ins = MeasureObjectIntensityDistribution()
measure_object_intensity_distribution_ins.add_setting(4, False, 100, True)
measure_object_intensity_distribution_record = measure_object_intensity_distribution_ins.run(image_instance, [objects])



final_result_dict = {}
if shape_feature_record is not None and len(shape_feature_record) > 0:
    final_result_dict.update(shape_feature_record)
if measure_object_intensity_record is not None and len(measure_object_intensity_record) > 0:
    final_result_dict.update(measure_object_intensity_record)
# if measure_granularity_record is not None and len(measure_granularity_record)>0:
#     final_result_dict.update(measure_granularity_record[0])
if measure_texture_record is not None and len(measure_texture_record) > 0:
    final_result_dict.update(measure_texture_record)
if measure_object_intensity_distribution_record is not None and len(
        measure_object_intensity_distribution_record) > 0:
    final_result_dict.update(measure_object_intensity_distribution_record)


final_result_df_temp=pd.DataFrame(final_result_dict)


# match bbx to wsi
final_result_df_temp=image_pack_instance.match_bbx_to_wsi(final_result_df_temp)

final_result_df_temp.to_csv(result_base_path+"/feature_result.csv")




