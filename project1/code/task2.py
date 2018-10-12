import numpy as np
import cv2
import math
from mycv import resize_shrink
import mynumpy as mnp
from task1 import eliminate_zero
import heapq
#from task1 import texture_filtering

SIGMAS = np.array([[1/np.sqrt(2), 1, np.sqrt(2), 2, 2*np.sqrt(2)],
                    [np.sqrt(2), 2, 2*np.sqrt(2), 4, 4*np.sqrt(2)],
                    [2*np.sqrt(2), 4, 4*np.sqrt(2), 8, 8*np.sqrt(2)],
                    [4*np.sqrt(2), 8, 8*np.sqrt(2), 16, 16*np.sqrt(2)]])


def texture_filtering(img_gray, kernel):
    """
    Notice:
        This is a almost same funcion as in task1. In task1 I have rewrite this funcion 
        so that it not use any  illegal numpy funcion mentioned in PIAZZA.
        However, in task2 if we use the task1 version texture_filtering, the program
        will be very slow in img_bank_gen step.
        So, here I use the version which use some basic numpy function. If you have any 
        concerns about here you can simply do these:
            1. delete this funcion
            2. uncomment #from task1 import texture_filtering
        Or let me know. My email is shaoming@buffalo.edu
        Thank you!
    
    Purpose:
        use to filter the gray image given the kernel
    Input:
        img_gray: 
            an two dimension ndarray matrix, dtype:usually is uint8 representint the gray image.
        kernel: 
            a two dimension ndarray matrix
    Output:
        The filtered image without padding around.
    """
    row_pad = math.floor(kernel.shape[0] / 2)
    col_pad = math.floor(kernel.shape[1] / 2)
    img_gray = np.pad(img_gray, ((row_pad,row_pad),(col_pad, col_pad)), 'constant')
    img_res = np.zeros(img_gray.shape)
    flipped_kernel = np.flip(kernel)
    for i in np.arange(row_pad, img_gray.shape[0] - row_pad):
        for j in np.arange(col_pad, img_gray.shape[1] - col_pad):
            img_res[i,j] = np.sum(img_gray[i-row_pad:i+row_pad+1, j-col_pad:j+col_pad+1] * flipped_kernel)
    return img_res[row_pad: img_res.shape[0] - row_pad, col_pad:img_res.shape[1] - col_pad]


def gaussin_val(x, y, sigma):
    """
    Purpose:
        Compute the gaussin val
    x:
        a real number 
    y:
        a real number
    sigma:
        a real number 
    """
    a = 1 / (2 * np.pi * mnp.power(sigma,2))
    b = np.exp(-(mnp.power(x,2) + mnp.power(y,2)) / (2 * mnp.power(sigma,2)))
    return a * b


def gaussin_kernel_gen(sigma, size=7):
    """
    Purpose: 
        compute the gaussin kernel given the sigma and kernel size
    Input:
        sigma: 
            a real number
        size: 
            int, the size of kernel
    Output:
        a gaussin kernel
    """
    
    if(size % 2 == 0):
        raise Exception("kernel size should be odd number")
    mat = np.asarray(mnp.zeros(size,size))
    pad = int(size/2)
    dividend = 0
    for i in range(size):
        for j in range(size):
            mat[i,j] = gaussin_val(j-pad, pad-i, sigma)
            dividend += mat[i,j]
    return mat / dividend


def kernels_db_gen(sigmas = SIGMAS):
    """
    Purpose:
        Generate a series of gaussin kernles given a array of sigmas
    Input:
        sigmas:
            a two dimension array which contains sigmas
    Output:
        a two dimension lists, each element is a kernel.
    """
    kernels = []
    for row in sigmas:
        mats = []
        for sigma in row:
            mats.append(gaussin_kernel_gen(sigma, 7))
        kernels.append(mats)
    return kernels


def resized_imgs_bank_gen(img_gray, layer):
    resized_imgs_bank = []
    for i in range(layer):
        img_resized = np.asarray(resize_shrink(img_gray, mnp.power(1/2,i), mnp.power(1/2,i)))
        resized_imgs_bank.append(img_resized)
    return resized_imgs_bank


def img_bank_gen(img_gray, kernels_db, resized_imgs_bank):
    """
    Purpose:
        Generate a series filtered image given the kernels database
    Input:
        img_gray: 
            a two dimension matrix representing the gray image, usually the dtype is uint8
        kernels_db: 
            a two dimension list, each elements is a kernel.
        resized_imgs_bank:
            a list contains resized_imgs
    Output:
        the img_bank, a two dimension list, each elements is a filterd image.
    """
    res = []
    print("in img_bank_gen")
    for i, row in enumerate(kernels_db):
        res_row = []
        img_resized = resized_imgs_bank[i]
        for kernel in row:
            res_row.append(texture_filtering(img_resized, kernel))
            print("fininsh a filterd img")
        print("row",i,"fininshed")
        res.append(res_row)
    return res


def dog_bank_gen(img_bank):
    """
    Purpose:
        Generate the Dog image for the images in img_bank
    Input:
        img_bank:
            a two dimension list, each elemetns is a filterd image.
    Output:
        res: a dog_bank, a two dimension list, each elements is a Dog image
    """
    
    res = []
    for row in img_bank:
        res_row = []
        for i in range(len(row[:-1])):
            res_row.append(row[i+1] - row[i])
        res.append(res_row)
    return res       


def check_min_max(upper_patch, patch, lower_patch):
    """
    Purpose:
        check if the middle pixel of patch is the maximum or the minimum pixel in the three patchs
    Input:
        Upper_patch:
        patch:
        lower_patch:
            each patch is a 3 by 3 two dimension matrix.
    Output: boolean
    """ 
    if ( (patch[1,1], 1) == mnp.min_all_count(patch) and patch[1,1] < mnp.min_all(upper_patch) 
            and patch[1,1] < mnp.min_all(lower_patch)
        or (patch[1,1],1) == mnp.max_all_count(patch) and patch[1,1] > mnp.max_all(upper_patch) 
            and patch[1,1] > mnp.max_all(lower_patch)):
        return True
    else:
        return False


def key_points_gen(img_upper, img, img_lower):
    """
    Purpose:
        Generate keypoints image
    Input:
        img_upper:
        img:
        img_lower:
           three gray images
    Output:
        res:
            a keypoints image in where the white pixels(255) are keypoints. 
    """
    
    res = []
    img_upper = np.ndarray.tolist(img_upper)
    img_upper = np.asarray(mnp.pad(img_upper,1,1,1,1))
    img = np.ndarray.tolist(img)
    img = np.asarray(mnp.pad(img,1,1,1,1))
    img_lower = np.ndarray.tolist(img_lower)
    img_lower = np.asarray(mnp.pad(img_lower,1,1,1,1))
    
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            upper_patch = img_upper[i-1:i+2, j-1:j+2]
            patch = img[i-1:i+2, j-1:j+2]
            lower_patch = img_lower[i-1:i+2, j-1:j+2]
            if check_min_max(upper_patch, patch, lower_patch):
                res.append((i-1,j-1))
    return res



def key_points_bank_gen(dog_bank):
    """
    Purpose:
        Generate the keypoints imgs bank by the dog_bank
    input:
        dog_bank:
            a two dimension list, each elements is a Dog image
    Output:
        key_points_imgs_bank:
            a two dimensions list, each element in the list is a keypoints image.
    """
    key_points_bank = []
    for i in range(len(dog_bank)):
        print("start new row")
        key_points_bank_row = []
        for j in range(1, len(dog_bank[i]) - 1):
            img_lower = dog_bank[i][j-1]
            img = dog_bank[i][j]
            img_upper = dog_bank[i][j+1]
            key_points_bank_row.append(key_points_gen(img_upper, img, img_lower))
            print("finish a key_points_list")
        key_points_bank.append(key_points_bank_row)
    return key_points_bank


def save_resized_imgs(resized_imgs_bank, show_img = False):
    """
    Input:
        resized_imgs_bank: 
            A list contains resized images
    Output:
        None
    """
    loc = "../task2_img" +"/resized_imgs/"
    for i in range(len(resized_imgs_bank)):
        name = "octave_" + str(i+1) + "_img" + ".jpg"
        cv2.imwrite(loc + name, resized_imgs_bank[i])
        if show_img:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, resized_imgs_bank[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def save_blured_imgs(img_bank, show_img = False):
    """
    Input:
        img_bank: 
            A two dimension list contains gaussin blured images
    Output:
        None
    """
    loc = "../task2_img" + "/blur_imgs/"
    for i in range(len(img_bank)):
        for j in range(len(img_bank[i])):
            name =  "octave_" + str(i+1) +"_blur_" + str(j+1) + "_img" + ".jpg"
            cv2.imwrite(loc + name, img_bank[i][j])
            if show_img:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.imshow(name, img_bank[i][j].astype(np.uint8))
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def save_dog_imgs(dog_bank, show_img = False):
    """
    Input:
        img_bank: 
            A two dimension list contains gaussin blured images
    Output:
        None
    """
    loc = "../task2_img" + "/dog_imgs/"
    for i in range(len(dog_bank)):
        for j in range(len(dog_bank[i])):
            name = "octave_" + str(i+1) +"_dog_" + str(j+1) + ".jpg"
            norm_dog_img = eliminate_zero(dog_bank[i][j])
            cv2.imwrite(loc + name,  norm_dog_img * 255)
            if show_img:
                cv2.namedWindow(loc + name, cv2.WINDOW_NORMAL)
                cv2.imshow(name, norm_dog_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def save_combined_key_points_imgs(key_points_bank, resized_imgs_bank, show_img = False):
    """
    Input:
        key_points_bank: 
            A two dimension list contains all key points indexs
        resized_imgs_bank:
            A one dimension list contain resized images
    Output:
        None
    """
    loc = "../task2_img" + "/combined_keypoints_imgs/"
    for i in range(len(key_points_bank)):
        resized_img = resized_imgs_bank[i]
        img_black = np.asarray(mnp.zeros(resized_img.shape[0], resized_img.shape[1]))
        set_pts = set()
        for j in range(len(key_points_bank[i])):
            set_pts = set_pts.union(set(key_points_bank[i][j]))
        name = "octave_" + str(i+1) +"_keypoints_img"
        img_clone = np.copy(resized_img)
        img_black_clone = np.copy(img_black)
        img_clone[[i for i in zip(*set_pts)]] = 255
        img_black_clone[[i for i in zip(*set_pts)]] = 255
        cv2.imwrite(loc + name + ".jpg", img_clone)
        cv2.imwrite(loc + name + "_black.jpg", img_black_clone)
        if show_img:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, img_clone)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def merge_key_points_bank(key_points_bank):
    res = set()
    for i in range(len(key_points_bank)):
        for val in key_points_bank[i]:
            res = res.union(set([(a * mnp.power(2,i), b * mnp.power(2,i)) for (a,b) in val]))
    return res


if __name__ == "__main__":
    img = cv2.imread("../task2_img/task2.jpg", 0)
    kernels_db = kernels_db_gen()
    resized_imgs_bank = resized_imgs_bank_gen(img, len(kernels_db))
    img_bank = img_bank_gen(img, kernels_db, resized_imgs_bank)
    dog_bank = dog_bank_gen(img_bank)
    key_points_bank = key_points_bank_gen(dog_bank)
    merged_key_points = merge_key_points_bank(key_points_bank)
    print("five left most points:(Consider the edge case)",
        [(b,a) for (a,b) in heapq.nsmallest(5,[(b,a) for (a,b) in merged_key_points])])
    five_left = []
    for val in merged_key_points:
        if val[1] == 1:
            five_left.append(val)
    five_left.sort()
    print("five left most points:(Not consider the edge case)", five_left[:5])

    save_resized_imgs(resized_imgs_bank, True)
    save_blured_imgs(img_bank, True)
    save_dog_imgs(dog_bank, True)
    save_combined_key_points_imgs(key_points_bank, resized_imgs_bank, True)











