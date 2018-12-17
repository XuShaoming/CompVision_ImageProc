import numpy as np
import cv2
import matplotlib.pyplot as plt
import task1
from mylibrary import count_pixels
import sys

def thresh(gray_img, T):
    res_img = np.zeros(gray_img.shape).astype(np.uint8)
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            if gray_img[i,j] > T:
                res_img[i,j] = 1
    return res_img

def numIslands(binary_img):
    if binary_img is None or binary_img.shape[0] == 0 or binary_img.shape[1] == 0:
        return 0, []
    
    obj_bank = []
    binary = binary_img.copy()
    
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            if binary[i,j] == 1:
                obj = []
                dfs(binary, i, j, obj)
                obj_bank.append(obj)
                
    return obj_bank

def dfs(grid, x, y, obj):
    n = grid.shape[0]
    m = grid.shape[1]
    if x < 0 or x > n - 1 or y < 0 or y > m - 1 or grid[x, y] == 0:
        return
    obj.append([x,y])
    grid[x,y] = 0
    dfs(grid, x+1, y, obj)
    dfs(grid, x-1, y, obj)
    dfs(grid, x, y-1, obj)
    dfs(grid, x, y+1, obj)


def draw_box(img, binary, obj_bank):
    res_img = img.copy()
    binary_color = cv2.cvtColor(binary*255, cv2.COLOR_GRAY2BGR)
    res_dcolor = binary_color
    
    for i, obj in enumerate(obj_bank):
        obj = np.asarray(obj).T
        up = min(obj[0])
        down = max(obj[0])
        left = min(obj[1])
        right = max(obj[1])
        print("object {}'s cordinate: left_up={}, right,down={}".format(i, (left, up), (right, down)))

        res_img = cv2.rectangle(res_img,(left, up),(right,down),(0,255,0),shift=0)
        res_dcolor = cv2.rectangle(res_dcolor,(left, up),(right,down),(0,255,0),shift=0)
        
    return res_img, res_dcolor 

if __name__ == "__main__":
    sys.setrecursionlimit(1500)
    img = cv2.imread("../task2b_img/segment.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixel_stat = count_pixels(img_gray)
    stat_list = np.asarray([[key,val] for key,val in pixel_stat.items() if key != 0]).T
    
    # save the histogram
    plt.bar(stat_list[0],stat_list[1],align='center') # A bar chart
    plt.title("pixel numbers count (value > 0)")
    plt.xlabel('pixel value')
    plt.ylabel('number')
    plt.savefig("../task2b_img/task2b_hist")
    plt.close()
    
    # observe the histogram to get T
    T = 205
    binary_img = thresh(img_gray, 205)
    #denoise the binary_img
    struc_elem = np.ones((3,3)).astype(np.uint8)
    denoised = task1.denoising(method=2)(binary_img, struc_elem)
    obj_bank = numIslands(denoised)
    print("There are {} objects".format(len(obj_bank)))
    for i in range(len(obj_bank)):
        print("object {} has {} numbers pixels".format(i, len(obj_bank[i])))
    
    img_res, dcolor_res = draw_box(img, denoised, obj_bank)
    cv2.imwrite('../task2b_img/res_segment.jpg', img_res)
    cv2.imwrite('../task2b_img/res_segment_2.jpg', dcolor_res)



