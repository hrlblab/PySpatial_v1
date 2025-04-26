# this file is to collect the image patch

class Patch:
    def __init__(self,input_image,input_mask,input_wsi_id,bbx):
        self.image=input_image
        self.mask=input_mask
        self.packed_id=None
        self.patch_id=input_wsi_id

        self.bbx=bbx #(x, y, x + w, y + h)
        # self.packed_id=None
        self.bbx_in_pack_img=None #(x, y, x + w, y + h)
        self.start_width=None
        self.start_height=None
        self.bbx_list_id=None


    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def area(self):
        return self.height*self.weight
