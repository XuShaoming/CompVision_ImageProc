import numpy as np
import cv2
import matplotlib.pyplot as plt
from mylibrary import count_pixels

def get_masked_img(img_gray):
    """
    Purpose:
        measure the weighted difference between the center point and its neighbors.
    Input:
        img_gray: matrix, value type can be real.
    Output:
        res_img: matrix, value type can be real.
    """
    mask = np.array([[-1,-1,-1],
                     [-1,8,-1],
                     [-1,-1,-1]])
    pad_img = np.pad(img_gray, ((1,1),(1,1)), 'constant')
    res_img = np.zeros(img_gray.shape)
    for i in range(1, pad_img.shape[0] - 1):
        for j in range(1, pad_img.shape[1] - 1):
            res_img[i-1,j-1] = np.sum(pad_img[i-1:i+2, j-1:j+2] * mask)
    return res_img

def detect_point(img_gray, T, white_only=False):
    """
    Purpose:
        Detect points in images
    Input:
        img_gray: matrix with value type real
        T: threshold, used to record pixels which value greater T.
        white_only: boolean, if False, detect black and white points. Otherwise, only detect white.
    Output:
        res_img: matrix with value type np.uint8. use to record points.
        stat: a dictionary, key is pixel value, value is number of given pixel value.
    """
    masked_img = get_masked_img(img_gray)
    if white_only == False:
        masked_img = np.abs(masked_img)
    
    stat = count_pixels(masked_img, include_border=False)
    res_img = np.zeros(img_gray.shape).astype(np.uint8)
    #ignore border
    for i in range(1, masked_img.shape[0] - 1):
        for j in range(1, masked_img.shape[1] - 1):
            if masked_img[i,j] > T:
                res_img[i,j] = 1
    return res_img, stat


def mark_img(img, binary_img):
    res = img.copy()
    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if binary_img[i,j] == 1:
                res[i,j] = np.array([0,0,255]).astype(np.uint8)
    return res

if __name__ == "__main__":
    ## detect both black and white points
    img = cv2.imread("../task2a_img/point.jpg")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary, stat = detect_point(gray_img, 110)
    res_img = mark_img(img, binary)
    key_val_arr = np.asarray([[key, val] for key,val in stat.items()])
    check_all = key_val_arr.T
    plt.bar(check_all[0],check_all[1],align='center') # A bar chart
    plt.title('task2_a all pixel (black white)')
    plt.xlabel('pixels value')
    plt.ylabel('numbers')
    plt.savefig("../task2a_img/task2a_hist_all_black_white")
    plt.close()

    check_greater = np.asarray([x for x in key_val_arr if x[0] > 100]).T
    plt.bar(check_greater[0],check_greater[1],align='center') # A bar chart
    plt.title('task2_a pixel val > 100 (black white)')
    plt.xlabel('pixels value')
    plt.ylabel('numbers')
    plt.savefig("../task2a_img/task2a_hist_100_black_white")
    plt.close()

    cv2.imwrite('../task2a_img/res_point.jpg', res_img)

    ## detect only white points
    gray_img = cv2.imread("../task2a_img/point.jpg", 0)
    binary, stat = detect_point(gray_img, 110, white_only=True)
    res_img = mark_img(img, binary)
    key_val_arr = np.asarray([[key, val] for key,val in stat.items()])
    check_all = key_val_arr.T
    plt.bar(check_all[0],check_all[1],align='center') # A bar chart
    plt.title('task2_a all pixel (white)')
    plt.xlabel('pixels value')
    plt.ylabel('numbers')
    plt.savefig("../task2a_img/task2a_hist_all_white")
    plt.close()

    check_greater = np.asarray([x for x in key_val_arr if x[0] > 100]).T
    plt.bar(check_greater[0],check_greater[1],align='center') # A bar chart
    plt.title('task2_a pixel val > 100 (white)')
    plt.xlabel('pixels value')
    plt.ylabel('numbers')
    plt.savefig("../task2a_img/task2a_hist_100_white")
    plt.close()

    cv2.imwrite('../task2a_img/res_point_white.jpg', res_img)