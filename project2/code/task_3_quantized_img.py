import numpy as np
import cv2
from io import StringIO
import task_3_k_mean as k_mean

UBIT = '50247057'
np.random.seed(sum([ord(c) for c in UBIT]))

def to_pixel_list(img):
    """
    Purpose:
        Transfer the two dimension image to a one dimension of pixels list.
    Input:
        img: a two dimension image. Both color and gray imgages are fine.
    Output:
        pixel_list: a list of pixel. value type float
    """
    pixel_list = np.empty((0,img[0].shape[1]))
    for row in img:
        pixel_list = np.append(pixel_list, row, axis=0)
    return pixel_list

def get_img(centers,shape):
    """
    Purpose:
        use the k_means results to generate images.
    Input:
        centers: a list of Center objects
        shape: the shape of the original image.
    Output: 
        the new image.
    """
    row_num, col_num, _ = shape
    new_img = np.empty(shape)
    for center in centers:
        pixel = center.center.astype(np.uint8)
        locs = list(map(lambda x: (int(x/col_num), int(x%col_num)) ,center.pts))
        for loc in locs:
            new_img[loc] = pixel
    return new_img.astype(np.uint8)

def quantized_img(img, k, init_fun = k_mean.init_centers_random, max_itr=10000, seed=20):
    """
    Prupose:
        Generate k mean images
    Input:
        img: a two dimension matrix. color or gray are fine.
        k: the number of colors.
        init_fun: functon, the funcion to init the centers for k mean algorithm. default: init_centers_random
        max_itr: int, the maximum number of iterations
        seed: use in init_fun
    Output:
        the new k mean image.
    """
    pixel_list = to_pixel_list(img)
    centers = k_mean.k_mean(pixel_list, k, init_fun, max_itr=100000, seed=seed)
    return get_img(centers,img.shape)

if __name__ == "__main__":
    UBIT = '50247057'
    seed = sum([ord(c) for c in UBIT])
    img = cv2.imread("../data/baboon.jpg")
    ks = [3,5,10,20]
    for k in ks:
        print("k =",k)
        new_img = quantized_img(img,k,seed=seed)
        cv2.imwrite("../task3_img/task3_baboon_"+str(k)+".jpg",new_img)
        print()





