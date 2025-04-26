import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import numpy as np
import cv2
from core.image_merged import Image_Pack
import PIL
import pandas as pd
import openslide

from feature.get_connect_component_object import GetConnectComponentObject
from feature.measure_object_size_shape import MeasureObjectSizeShape
from feature.measure_texture import  MeasureTexture
from feature.measure_object_intensity import MeasureObjectIntensity
from feature.measure_object_intensity_distribution import MeasureObjectIntensityDistribution


#The final code for the EI paper

result_root_path="example/result/KPMP_result/"

data_root_path="example/data/KPMP/"
#read the origin WSI # origin_img.png
origin_img_path=data_root_path+"img/S-1908-000799_PAS_1of2.svs"

wsi_img_slide = openslide.OpenSlide(origin_img_path)
slide_width = int(wsi_img_slide.properties["openslide.level[0].width"])
slide_height = int(wsi_img_slide.properties["openslide.level[0].height"])

wsi_img= wsi_img_slide.read_region((0, 0), 0, (slide_width, slide_height))
wsi_img = wsi_img.convert("L")
wsi_img=np.array(wsi_img)


# create the folder to save the result
os.makedirs(result_root_path, exist_ok=True)

# define a dataframe to save the time


for i in [1,2,4]:
    add_img_time=0
    get_obj_time=0
    measure_size_shape_time=0
    measure_texture_time=0
    measure_intensity_time=0
    measure_intensity_distribution_time=0
    final_final_result_dict=None
    # temp to save time



    #read the mask
    mask_path=data_root_path+"/mask/"+str(i)+".npy"
    mask_img=np.load(mask_path).astype(np.int32)

    # pack the image
    image_pack_instance = Image_Pack()
    image_pack_instance = image_pack_instance.get_image_patch_with_senatic_label(mask_img, wsi_img, image_pack_instance)
    del mask_img


    # merge patch
    # image_pack_instance.pack_rectangles(3000, 3000)

    flag=0
    for temp_patch in image_pack_instance.patchls:

        shape_feature_record=None
        measure_texture_record=None
        measure_object_intensity_distribution_record=None
        measure_object_intensity_record=None


        # calculate the feature

        # add to image class
        from core.image import Image

        image_instance = Image(
            temp_patch.image,
            mask=temp_patch.mask.astype('bool'),
            path_name=None,
            file_name=None,
            scale=None,
            channelstack=temp_patch.image.ndim == 3 and temp_patch.image.shape[
                -1] > 3,
            # uint8_image=None,
            convert=False,
        )


        # identify primary object
        get_connect_ins = GetConnectComponentObject()
        objects = get_connect_ins.run(temp_patch.mask.astype(np.uint8), image_instance)



        # measure object size shape
        measure_size_shape_ins = MeasureObjectSizeShape([objects])
        measure_size_shape_ins.add_setting(calculate_advanced_input=True, calculate_zernikes_input=True)
        shape_feature_record = measure_size_shape_ins.run()


        # measure object texture
        measure_texture_ins = MeasureTexture(False)
        measure_texture_ins.add_setting("Objects", 256, 2, [image_instance])
        measure_texture_ins.add_object([objects])
        measure_texture_record = measure_texture_ins.run()

        # print("finish measure texture")

        # measure object intensity

        measure_object_intensity_ins = MeasureObjectIntensity([image_instance], [objects])
        measure_object_intensity_record = measure_object_intensity_ins.run()

        # print("finish measure intensity")


        measure_object_intensity_distribution_ins = MeasureObjectIntensityDistribution()
        measure_object_intensity_distribution_ins.add_setting(4, False, 100, True)
        measure_object_intensity_distribution_record = measure_object_intensity_distribution_ins.run(image_instance,
                                                                                                     [objects])

        # update time

        final_result_dict_temp = {}
        #final_result_dict_temp["merge_image_id"]=merge_image_id
        if shape_feature_record is not None and len(shape_feature_record) > 0:
            final_result_dict_temp.update(shape_feature_record)
        if measure_object_intensity_record is not None and len(measure_object_intensity_record) > 0:
            final_result_dict_temp.update(measure_object_intensity_record)
        # if measure_granularity_record is not None and len(measure_granularity_record)>0:
        #     final_result_dict.update(measure_granularity_record[0])
        if measure_texture_record is not None and len(measure_texture_record) > 0:
            final_result_dict_temp.update(measure_texture_record)
        if measure_object_intensity_distribution_record is not None and len(
                measure_object_intensity_distribution_record) > 0:
            final_result_dict_temp.update(measure_object_intensity_distribution_record)
        # if measure_object_neighbors_record is not None and len(measure_object_neighbors_record) > 0:
        #     final_result_dict.update(measure_object_neighbors_record)

        final_result_dict_temp = pd.DataFrame(final_result_dict_temp)
        final_result_dict_temp['object_id']=flag
        flag=flag+1

        # concat to the final_final_result
        if final_final_result_dict is None:
            final_final_result_dict=final_result_dict_temp
        else:
            final_final_result_dict = pd.concat([final_final_result_dict, final_result_dict_temp], axis=0, ignore_index=True)


    final_final_result_dict.to_csv(result_root_path+"/"+str(i)+".csv")

