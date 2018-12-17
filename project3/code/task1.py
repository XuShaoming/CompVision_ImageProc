import numpy as np
import cv2
import math

def dilation(binary_image, selem):
    """
    Purpose:
        Do dilation operation on binary image.
    Input:
        binary_image: int, 2D numpy array, value 0 or 1
        selem: int, 2D numpy array. The structuring element
    Output:
        the dilated image.
    """
    row_pad = math.floor(selem.shape[0] / 2)
    col_pad = math.floor(selem.shape[1] / 2)
    res_img = np.zeros(binary_image.shape)
    check_img = np.pad(binary_image, ((row_pad,row_pad),(col_pad, col_pad)), 'constant')
    flipped_selem = np.flip(selem)
    for i in np.arange(row_pad, check_img.shape[0] - row_pad):
        for j in np.arange(col_pad, check_img.shape[1] - col_pad):
            patch = check_img[i-row_pad:i+row_pad+1, j-col_pad:j+col_pad+1]
            if np.any(patch & flipped_selem == 1): 
                res_img[i-row_pad, j-col_pad] = 1            
    return res_img.astype(np.uint8)

def erosion(binary_image, selem):
    """
    Purpose:
        Do erosion operation on binary image.
    Input:
        binary_image: int, 2D numpy array, value 0 or 1
        selem: int, 2D numpy array. The structuring element
    Output:
        the eroded image.
    """
    row_pad = math.floor(selem.shape[0] / 2)
    col_pad = math.floor(selem.shape[1] / 2)
    res_img = np.zeros(binary_image.shape)
    check_img = np.pad(binary_image, ((row_pad,row_pad),(col_pad, col_pad)), 'constant')
    for i in np.arange(row_pad, check_img.shape[0] - row_pad):
        for j in np.arange(col_pad, check_img.shape[1] - col_pad):
            patch = check_img[i-row_pad:i+row_pad+1, j-col_pad:j+col_pad+1]
            if np.all(patch & selem == selem):
                res_img[i-row_pad, j-col_pad] = 1
    return res_img.astype(np.uint8)

def opening(binary_image, selem):
    return dilation(erosion(binary_image, selem), selem)

def closing(binary_image, selem):
    return erosion(dilation(binary_image, selem), selem)

def boundary(binary_image):
    selem = np.ones((3,3)).astype(np.uint8)
    return binary_image - erosion(binary_image, selem)

def denoising(method=1):
    def method_1(binary_image, selem):
        return closing(opening(binary_image, selem), selem)
    
    def method_2(binary_image, selem):
        return opening(closing(binary_image, selem), selem)
    
    if method == 1:
        return method_1
    else:
        return method_2

def threshold(gray_img, thresh):
    res_img = np.zeros(gray_img.shape)
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            if gray_img[i,j] > thresh:
                res_img[i,j] = 1
    return res_img.astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread("../task1_img/noise.jpg")
    img = cv2.imread("../task1_img/noise.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = 127
    struc_elem = np.ones((3,3)).astype(np.uint8)
    binary = threshold(gray, thresh)

    print("**** task1 question a ****")
    res_noise1 = denoising(method=1)(binary, struc_elem)
    name = "../task1_img" + "/res_noise1" + ".jpg"
    cv2.imwrite(name, res_noise1*255)
    res_noise2 = denoising(method=2)(binary, struc_elem)
    name = "../task1_img" + "/res_noise2" + ".jpg"
    cv2.imwrite(name, res_noise2*255)

    print("**** task1 question c ****")
    res_bound1 = boundary(res_noise1)
    name = "../task1_img" + "/res_bound1" + ".jpg"
    cv2.imwrite(name, res_bound1*255)
    res_bound2 = boundary(res_noise2)
    name = "../task1_img" + "/res_bound2" + ".jpg"
    cv2.imwrite(name, res_bound2*255)




