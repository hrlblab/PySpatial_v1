import cv2
# this file is for the merged image
import core.objects
import numpy as np
from core.patch import Patch
from shapely.geometry import Point, box
from rtree import index
import pandas as pd


class Image_Pack:
    def __init__(self):
        self.patchls=[]
        self.merged_image=[]
        self.merged_mask=[]
        self.bbx_list=None

        self.idx=index.Index() # to match the patch to origin wsi
        self.idx_ls=[] #This is for the KPMP data


    def addpatch(self,input_patch):
        self.patchls.append(input_patch)

    def normalize_mask(self,unnormalized_mask):
        binary_mask = unnormalized_mask.astype(float) / 255
        binary_mask[binary_mask < 0.2] = 0
        binary_mask[binary_mask > 0.8] = 1
        return binary_mask

    def get_connect_label(self,uint8_image):
        #gray_uint8_img = cv2.cvtColor(uint8_image, cv2.COLOR_BGR2GRAY)
        #gray_uint8_img=uint8_image.copy()
        gray_uint8_img = cv2.threshold(uint8_image, 0, 255, cv2.THRESH_OTSU)[1]
        num_labels, labels_im = cv2.connectedComponents(gray_uint8_img, connectivity=8)

        return num_labels, labels_im


    # this is the pack function for the EI paper
    def get_boundingbox_of_isolated_object_with_semantic_mask(self,mask_img,wsi_img,image_pack_instance):
        labeled_image = mask_img
        num_labels=np.unique(labeled_image)
        bounding_boxes = {}
        for i in num_labels:
            if i == 0:
                continue  # 跳过背景

            # get the pixel of current label
            points = np.column_stack(np.where(labeled_image == i))

            # use OpenCV to find bounding box
            x, y, w, h = cv2.boundingRect(points)
            bounding_boxes[i] = (x, y, x + w, y + h)

            uint8_patch=mask_img[ int(x) - 3:int(x + w) + 3,int(y) - 3:int(y + h) + 3]
            uint8_img=(uint8_patch==i).astype("uint8")

            # add the mask and wsi patch to the pack instance
            image_pack_instance.addpatch(Patch(
                wsi_img[ int(x) - 3:int(x + w) + 3,int(y) - 3:int(y + h) + 3].astype(np.float64)/ 255,
                uint8_img ,i,(x, y, x + w, y + h)))


        image_pack_instance.bbx_list=bounding_boxes


        return image_pack_instance




    def get_boundingbox_of_isolated_object(self,uint8_img,wsi_img,image_pack_instance,wsi_id):
        uint8_img=uint8_img.astype("uint8")
        num_labels, labeled_image = self.get_connect_label(uint8_img)
        bounding_boxes = {}
        for i in range(num_labels):
            if i == 0:
                continue  # 跳过背景

            # get the pixel of current label
            points = np.column_stack(np.where(labeled_image == i))

            # use OpenCV to find bounding box
            x, y, w, h = cv2.boundingRect(points)
            bounding_boxes[i] = (x, y, x + w, y + h)

            # add the mask and wsi patch to the pack instance
            image_pack_instance.addpatch(Patch(
                wsi_img[ int(y) - 3:int(y+h + 3), int(x) - 3:int(x+w + 3)].astype(np.float64)/ 255,
                uint8_img[ int(y) - 3:int(y+h) + 3, int(x) - 3:int(x+w) + 3] ,wsi_id,(x, y, x + w, y + h)))

        image_pack_instance.bbx_list=bounding_boxes


        return image_pack_instance



    # this is the pack function for the EI paper
    def get_image_patch_with_senatic_label(self,mask_img,wsi_img,image_pack_instance):
        # use the connect componet to identiy object
        image_pack_instance = self.get_boundingbox_of_isolated_object_with_semantic_mask(mask_img,wsi_img,image_pack_instance)

        return image_pack_instance



    def get_image_patch(self,mask_img,wsi_img,image_pack_instance,wsi_id):
        mask_img_normalized=self.normalize_mask(mask_img)
        # use the connect componet to identiy object
        image_pack_instance = self.get_boundingbox_of_isolated_object(mask_img_normalized,wsi_img,image_pack_instance,wsi_id)

        return image_pack_instance

    def find_bbx_for_point(self,x, y):
        point = Point(x, y)
        possible_matches = list(self.idx.intersection((x, y, x, y)))
        for i in possible_matches:
            # print("************************************")
            # print(len(possible_matches))
            # print(possible_matches)
            # print(self.patchls[i].bbx)
            # print([x,y])

            x1, y1, x2, y2 = self.patchls[i].bbx_in_pack_img
            if box(x1, y1, x2, y2).contains(point):
                return self.patchls[i].bbx
        return None, None, None, None  # 若无匹配返回空值


    def find_bbx_for_point_for_multi_page(self,x, y,merge_image_id):
        point = Point(x, y)
        possible_matches = list(self.idx_ls[merge_image_id].intersection((x, y, x, y)))
        for i in possible_matches:
            # print("************************************")
            # print(len(possible_matches))
            # print(possible_matches)
            # print(self.patchls[i].bbx)
            # print([x,y])

            x1, y1, x2, y2 = self.patchls[i].bbx_in_pack_img
            if box(x1, y1, x2, y2).contains(point):
                return self.patchls[i].bbx
        print("Do not match")
        return None, None, None, None  # 若无匹配返回空值



    def match_bbx_to_wsi(self,feature_df):
        # create R-tree index
        # idx = index.Index()
        # patch_mapping = {}

        # for i, patch in enumerate(self.patchls):
        #     x1, y1, x2, y2 = patch.bbx
        #     idx.insert(i, (x1, y1, x2, y2))
            # patch_mapping[i] = patch.packed_id

        # self.idx=idx
        # match the center to the bbx
        feature_df[['bbx_x1', 'bbx_y1', 'bbx_x2', 'bbx_y2']] = feature_df.apply(
            lambda row: pd.Series(self.find_bbx_for_point(row['AreaShape_Center_X'], row['AreaShape_Center_Y'])),
            axis=1
        )

        feature_df['bbx_x1']=feature_df['bbx_x1']+4
        feature_df['bbx_y1'] = feature_df['bbx_y1'] + 4
        feature_df['bbx_x2'] = feature_df['bbx_x2'] -3
        feature_df['bbx_y2'] = feature_df['bbx_y2'] -3


        feature_df["AreaShape_Center_X"]=feature_df["AreaShape_Center_X"]+feature_df["bbx_x1"]
        feature_df["AreaShape_Center_Y"] = feature_df["AreaShape_Center_Y"] + feature_df["bbx_y1"]

        feature_df['Location_CenterMassIntensity_X_podocyte_nuclei']=feature_df['Location_CenterMassIntensity_X_podocyte_nuclei']+feature_df["bbx_x1"]
        feature_df['Location_CenterMassIntensity_Y_podocyte_nuclei']=feature_df['Location_CenterMassIntensity_Y_podocyte_nuclei']+ feature_df["bbx_y1"]
        feature_df['Location_MaxIntensity_X_podocyte_nuclei']=feature_df['Location_MaxIntensity_X_podocyte_nuclei']+feature_df["bbx_x1"]
        feature_df['Location_MaxIntensity_Y_podocyte_nuclei']=feature_df['Location_MaxIntensity_Y_podocyte_nuclei']+ feature_df["bbx_y1"]



        feature_df.drop(columns=['AreaShape_BoundingBoxMaximum_X', 'AreaShape_BoundingBoxMaximum_Y','AreaShape_BoundingBoxMinimum_X','AreaShape_BoundingBoxMinimum_Y'], inplace=True)
        return feature_df


    def match_bbx_to_wsi_for_multi_page(self,feature_df,merge_image_id):
        # create R-tree index
        # idx = index.Index()
        # patch_mapping = {}

        # for i, patch in enumerate(self.patchls):
        #     x1, y1, x2, y2 = patch.bbx
        #     idx.insert(i, (x1, y1, x2, y2))
            # patch_mapping[i] = patch.packed_id

        # self.idx=idx
        # match the center to the bbx
        feature_df[['bbx_x1', 'bbx_y1', 'bbx_x2', 'bbx_y2']] = feature_df.apply(
            lambda row: pd.Series(self.find_bbx_for_point_for_multi_page(row['AreaShape_Center_X'], row['AreaShape_Center_Y'],merge_image_id)),
            axis=1
        )

        feature_df['bbx_x1']=feature_df['bbx_x1']+4
        feature_df['bbx_y1'] = feature_df['bbx_y1'] + 4
        feature_df['bbx_x2'] = feature_df['bbx_x2'] -3
        feature_df['bbx_y2'] = feature_df['bbx_y2'] -3


        feature_df["AreaShape_Center_X"]=feature_df["AreaShape_Center_X"]+feature_df["bbx_x1"]
        feature_df["AreaShape_Center_Y"] = feature_df["AreaShape_Center_Y"] + feature_df["bbx_y1"]

        feature_df['Location_CenterMassIntensity_X_podocyte_nuclei']=feature_df['Location_CenterMassIntensity_X_podocyte_nuclei']+feature_df["bbx_x1"]
        feature_df['Location_CenterMassIntensity_Y_podocyte_nuclei']=feature_df['Location_CenterMassIntensity_Y_podocyte_nuclei']+ feature_df["bbx_y1"]
        feature_df['Location_MaxIntensity_X_podocyte_nuclei']=feature_df['Location_MaxIntensity_X_podocyte_nuclei']+feature_df["bbx_x1"]
        feature_df['Location_MaxIntensity_Y_podocyte_nuclei']=feature_df['Location_MaxIntensity_Y_podocyte_nuclei']+ feature_df["bbx_y1"]



        feature_df.drop(columns=['AreaShape_BoundingBoxMaximum_X', 'AreaShape_BoundingBoxMaximum_Y','AreaShape_BoundingBoxMinimum_X','AreaShape_BoundingBoxMinimum_Y'], inplace=True)
        return feature_df





    def pack_rectangles(self,max_width_total,max_height_total):

        current_x, current_y = 0, 0
        max_height_in_row = 0

        temp_packed_image=np.zeros((max_width_total,max_height_total))
        temp_packed_mask = np.zeros((max_width_total, max_height_total))
        packed_id=0
        for patch in self.patchls:
            # print("patch_id"+str(patch.patch_id))
            width=patch.width
            height=patch.height
            # if the width is reach limited
            if (current_x + width >= max_width_total) or (current_y+height>=max_height_total):
                current_x = 0
                # if the height is reach limited
                if current_y+max_height_in_row+height>=max_height_total:
                    current_y=0
                    packed_id=packed_id+1
                    max_height_in_row = 0
                    # the image is full,add the temp image and mask to the list
                    self.merged_image.append(temp_packed_image)
                    self.merged_mask.append(temp_packed_mask)

                    # get the new temp mask and image
                    temp_packed_image = np.zeros((max_width_total, max_height_total))
                    temp_packed_mask = np.zeros((max_width_total, max_height_total))

                else:
                    # change to a new line
                    current_y += max_height_in_row
                    max_height_in_row = 0

            # copy the area to the merged mask and merged image
            patch.packed_id=packed_id
            patch.start_height=current_y
            patch.start_width=current_x
            temp_packed_mask[current_y:current_y+height,current_x:current_x+width]=patch.mask.copy()
            temp_packed_image[ current_y:current_y+height,current_x:current_x+width] =patch.image.copy()

            # update the idx: to make sure find the final coordinate
            self.idx.insert(patch.patch_id, (current_x, current_y, current_x+width, current_y+height))
            patch.bbx_in_pack_img=[current_x, current_y, current_x+width, current_y+height]


            #update the height and width
            current_x += width
            max_height_in_row = max(max_height_in_row, height)

        # add the final temp mask and image to the list
        self.merged_image.append(temp_packed_image)
        self.merged_mask.append(temp_packed_mask)

        # return packed_positions



    def pack_rectangles_for_multi_page(self,max_width_total,max_height_total):
        # patchls_new=self.patchls.copy()
        #sort the patchls from big to small
        # patchls_new.sort(key=lambda  x: x.height*x.width, reverse=True)
        # pack the rectangle
        # packed_positions = []
        current_x, current_y = 0, 0
        max_height_in_row = 0
        temp_idx=index.Index()

        temp_packed_image=np.zeros((max_width_total,max_height_total))
        temp_packed_mask = np.zeros((max_width_total, max_height_total))
        packed_id=0
        patch_id=-1
        for patch in self.patchls:
            patch_id=patch_id+1
            # print("patch_id"+str(patch.patch_id))
            width=patch.width
            height=patch.height
            # if the width is reach limited
            if (current_x + width >= max_width_total) or (current_y+height>=max_height_total):
                current_x = 0
                # if the height is reach limited
                if current_y+max_height_in_row+height>=max_height_total:
                    current_y=0
                    packed_id=packed_id+1
                    max_height_in_row = 0
                    # the image is full,add the temp image and mask to the list
                    self.merged_image.append(temp_packed_image)
                    self.merged_mask.append(temp_packed_mask)
                    #change to the new temp index
                    self.idx_ls.append(temp_idx)
                    temp_idx=index.Index()


                    # get the new temp mask and image
                    temp_packed_image = np.zeros((max_width_total, max_height_total))
                    temp_packed_mask = np.zeros((max_width_total, max_height_total))

                else:
                    # change to a new line
                    current_y += max_height_in_row
                    max_height_in_row = 0

            # copy the area to the merged mask and merged image
            patch.packed_id=packed_id
            patch.start_height=current_y
            patch.start_width=current_x
            patch.patch_id=patch_id
            temp_packed_mask[current_y:current_y+height,current_x:current_x+width]=patch.mask.copy()
            temp_packed_image[ current_y:current_y+height,current_x:current_x+width] =patch.image.copy()

            # update the idx: to make sure find the final coordinate
            temp_idx.insert(patch.patch_id, (current_x, current_y, current_x+width, current_y+height))
            patch.bbx_in_pack_img=[current_x, current_y, current_x+width, current_y+height]


            #update the height and width
            current_x += width
            max_height_in_row = max(max_height_in_row, height)

        # add the final temp mask and image to the list
        self.merged_image.append(temp_packed_image)
        self.merged_mask.append(temp_packed_mask)
        self.idx_ls.append(temp_idx)

        # return packed_positions
