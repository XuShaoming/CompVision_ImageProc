import cv2
import numpy as np
import math
import os
import re


METHODS = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


def preproces_laplacian(img):
    blur_img = cv2.GaussianBlur(img,(3,3),0)
    return cv2.Laplacian(blur_img,cv2.CV_8U)

def preprocess_template(template):
    
    ## elimate the not useful information
    index_top = 0
    index_bottom = template.shape[0] - 1
    index_left = 0
    index_right = template.shape[1] - 1
    for row in template:
        if max(row) > 100:
            break
        index_top += 1
    
    for row in reversed(template):
        if max(row) > 100:
            break
        index_bottom -= 1
        
    for col in template.T:
        if max(col) > 100:
            break
        index_left += 1
        
    for col in reversed(template.T):
        if max(col) > 100:
            break
        index_right -= 1
    
    res = template[index_top : index_bottom + 1, index_left : index_right + 1]
        
    return res

def template_match(loc="../task3_bonus/", temp_name="template_1.jpg", img_prefix="t1_",num=6,meth = 'cv2.TM_CCORR_NORMED', has_mask = False):
    """
    Input:
        meth: String
            The names of template matching method
        has_mask: boolean
            True: use mask on template. Only support TM_CCORR_NORMED and TM_SQDIFF
            False: not use
    """
    m = re.search(r'cv2.(\w+)', meth)
    save_loc = loc + m.group(1)
    
    template = cv2.imread(loc + temp_name,0)
    mask_ = None
    if has_mask:
        mask_ = np.ones(template.shape,dtype=np.uint8)
        mask_[template < 80] = 0
        save_loc = save_loc + "/mask"
    
    method = eval(meth)
    #preproces_laplacian
    template = preproces_laplacian(template)
    
    try:
        os.makedirs(save_loc)
    except FileExistsError:
        print("use existing folder:", save_loc)
    
    for i in range(1,num+1):
        name = img_prefix + str(i) + ".jpg"
        img = cv2.imread(loc + name)
        img_gray = cv2.imread(loc + name, 0)
        img_gray = preproces_laplacian(img_gray)
        h, w = template.shape

        # Apply template Matching
        res = cv2.matchTemplate(img_gray,template,method, mask = mask_)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img,top_left, bottom_right, (0,0,255), 2)
        cv2.imwrite(save_loc + "/" + img_prefix + str(i) + "_res" + ".jpg", img)


if __name__ == "__main__":
    template_match("../task3_img/",'template_1.jpeg',"pos_", 15,'cv2.TM_CCORR_NORMED', False)
    template_match("../task3_img/",'template_1.jpeg',"pos_", 15, 'cv2.TM_CCOEFF_NORMED', False)
    template_match("../task3_img/",'template_1.jpeg',"pos_", 15, 'cv2.TM_SQDIFF_NORMED', False)
    ##For bonus part:
    template_match("../task3_bonus/", "template_1.jpg", "t1_", 6)
    template_match("../task3_bonus/", "template_2.jpg", "t2_", 6)
    template_match("../task3_bonus/", "template_3.jpg", "t3_", 6)




