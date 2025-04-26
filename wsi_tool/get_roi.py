import cv2
from core.patch import Patch
from core.image_merged import Image_Pack
import numpy as np

def get_connect_label(self, uint8_image):
    # the input should between 0 and 255
    # gray_uint8_img = cv2.cvtColor(uint8_image, cv2.COLOR_BGR2GRAY)
    gray_uint8_img = uint8_image.copy()
    gray_uint8_img = cv2.threshold(gray_uint8_img, 0, 255, cv2.THRESH_OTSU)[1]
    num_labels, labels_im = cv2.connectedComponents(gray_uint8_img, connectivity=8)

def binarize_mask(mask,threshold=128):
    _,binary_mask = cv2.threshold(mask,threshold,255,cv2.THRESH_BINARY)
    return binary_mask


def find_bounding_boxes(mask,img,image_pack_instance,wsi_id):
    num_labels,labels,stats,centroids=cv2.connectedComponentsWithStats(mask, connectivity=8)
    # add each to the patch object
    for i in range(1,num_labels):
        x,y,w,h = stats[i]
        image_pack_instance.addpatch(Patch(np.array(
            img.crop((int(x) - 2, int(y) - 2, int(x+w) + 2, int(y+h) + 2)).convert(
                mode="L"), dtype=np.float64) / 255, np.array(
            mask.crop((int(x) - 2, int(y) - 2, int(x+w) + 2, int(y+h) + 2)),
            dtype=np.uint8) / 255, wsi_id))

    return image_pack_instance



def get_roi(wsi_img,wsi_mask,wsi_id):
    # binary_mask
    binary_mask=binarize_mask(wsi_img)
    # get image patch
    image_pack_instance=Image_Pack()
    image_pack_instance=find_bounding_boxes(binary_mask,wsi_img,image_pack_instance,wsi_id)
    return image_pack_instance

