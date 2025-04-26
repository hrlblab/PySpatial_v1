import os
import numpy as np
import pandas as pd
import PIL
import cv2
import time

def find_base_name(wsi_folder_path):
    # this is to find the base name of the file in the image(end with svs)
    for file_name in os.listdir(wsi_folder_path):
        # 检查文件是否以 .svs 结尾
        if file_name.endswith('.svs'):
            # 去掉文件的后缀名
            base_name = os.path.splitext(file_name)[0]
    return base_name


def add_border(patch, border_width=5):
    rows, cols = patch.shape
    bordered_patch = np.zeros((rows + 2 * border_width, cols + 2 * border_width), dtype=patch.dtype)
    bordered_patch[border_width:border_width + rows, border_width:border_width + cols] = patch
    return bordered_patch

def save_mask(mask,path,obj_id):
    PIL.Image.fromarray((mask * (255/np.max(mask))).astype('uint8'), mode='L').save(path+str(obj_id)+".png")

def save_origin(img,path,obj_id):
    cv2.imwrite(path+str(obj_id)+".png", img)

def mask_img(mask,img):
    img_cp=img.copy()
    img_cp[:, :, 0] = img_cp[:, :, 0]*mask
    img_cp[:, :, 1] = img_cp[:, :, 1]*mask
    img_cp[:, :, 2] = img_cp[:, :, 2]*mask
    return img_cp



def extract_patches_with_border(mask, coordinate_df,origin_img_np,patch_mask_result_path,patch_img_result_path,patch_mask_img_result_path,coord_csv_path,start_time1,start_time2):
    #border_width = 3
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids != 0]  # 移除背景ID（0）

    for obj_id in object_ids:
        # 找出这个object在mask中的位置
        positions = np.where(mask == obj_id)
        ymin, xmin = np.min(positions[0]), np.min(positions[1])
        ymax, xmax = np.max(positions[0]), np.max(positions[1])

        ymax=ymax+50
        xmax=xmax+50
        ymin=ymin-50
        xmin=xmin-50

        # 裁剪这个object
        patch_temp = mask[ymin:ymax, xmin:xmax]

        patch=patch_temp.copy()

        patch[patch!= obj_id]=0
        patch[patch == obj_id] = 1

        # 给patch加边界
        #mask_patch = add_border(patch, border_width)
        mask_patch=patch

        # 保存加边界的patch和位置
        true_ymin=ymin
        true_xmin=xmin
        true_ymax=ymax
        true_xmax=xmax


        # get origin image patch
        patch_origin_image = origin_img_np[true_ymin:true_ymax, true_xmin:true_xmax]


        # save the result
        # mask
        save_mask(mask_patch,patch_mask_result_path,obj_id)
        # origin image
        save_origin(patch_origin_image,patch_img_result_path,obj_id)
        # mask and origin
        mask_origin_patch=mask_img(mask_patch,patch_origin_image)
        save_origin(mask_origin_patch,patch_mask_img_result_path,obj_id)

        # add the coord
        new_row_dic = {coordinate_df.columns[0]: [obj_id],
                       coordinate_df.columns[1]: [true_xmin],
                       coordinate_df.columns[2]: [true_ymin],
                       coordinate_df.columns[3]: [true_xmax],
                       coordinate_df.columns[4]: [true_ymax]}
        new_row = pd.DataFrame(new_row_dic, columns=coordinate_df.columns)
        coordinate_df = pd.concat([coordinate_df, new_row], ignore_index=True)


    end_time=time.time()
    exec_time1=end_time-start_time1
    exec_time2=end_time-start_time2

    coordinate_df["time1_include_create_folder"]=exec_time1
    coordinate_df["time2"] = exec_time2

    # save the coord_csv
    coordinate_df.to_csv(coord_csv_path,index=False)

    return coordinate_df


