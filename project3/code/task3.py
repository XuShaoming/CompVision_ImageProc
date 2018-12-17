import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mylibrary import count_pixels
import task1

## the edge detection part
# sober for edge detection
VERTICAL_SOBEL_3BY3 = np.array([[-1,2,-1],
                          [-1,2,-1],
                          [-1,2,-1]])

HORIZONTAL_SOBEL_3BY3 = np.array([[-1,-1,-1],
                         [2,2,2],
                         [-1,-1,-1]])
POS45_SOBEL_3BY3 = np.array([[-1,-1,2],
                         [-1,2,-1],
                         [2,-1,-1]])
NEG45_SOBEL_3BY3 = np.array([[2,-1,-1],
                         [-1,2,-1],
                         [-1,-1,2]])

def texture_filtering(img_gray, kernel):
    """
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
    
    res_img = np.zeros(img_gray.shape)
    check_img = np.pad(img_gray, ((row_pad,row_pad),(col_pad, col_pad)), 'constant')
    flipped_kernel = np.flip(kernel)
    
    for i in range(row_pad, check_img.shape[0] - row_pad):
        for j in range(col_pad, check_img.shape[1] - col_pad):
            patch = check_img[i-row_pad:i+row_pad+1, j-col_pad:j+col_pad+1]
            res_img[i-row_pad, j-col_pad] = np.sum(patch * flipped_kernel)
    return res_img


def eliminate_zero(img, method=1):
    """
    Purpose:
        two ways to eliminate the negative value or the value out of 255.
    Input:
        img: two dimension matrix
            the raw image. dtype usually is float64 with pixel < 0 or pixel > 255
        method: int
            default is 1 which directs to first method
            the 2 will direct to the second method.
    Output:
        a matrix dtype range zero to one. 
    """
    if method == 1:
        min_ = np.min(img)
        max_ = np.max(img)
        return (img - min_) / (max_ - min_)
    elif method == 2:
        abs_img = np.abs(img)
        return abs_img / np.max(abs_img)
    else :
        print("method is 1 or 2")


def combine_edge(hori,vert,pos45,neg45,T):
    """
    Purpose:
        to combine edges in four direction. 
        Mark result image 1 in given loc if any of these four edge images pixels absolute value
        greater than T in that loc.
    Output:
        res_img: binary img, value type np.uint8
    """
    hori = np.abs(hori)
    vert = np.abs(vert)
    pos45 = np.abs(pos45)
    neg45 = np.abs(neg45)
    res_img = np.zeros(hori.shape).astype(np.uint8)
    for i in range(hori.shape[0]):
        for j in range(hori.shape[1]):
            if hori[i,j] > T or vert[i,j] > T or pos45[i,j] > T or neg45[i,j] > T :
                res_img[i,j] = 1
    return res_img

def ignore_border(img_gray):
    """
    Purpose:
        ignore the img bodre by setting value as 0.
        notice: Affect original img.
    Output:
        the pointer point to original image.
    """
    img_gray[0] = 0
    img_gray[-1] = 0
    for row in img_gray:
        row[0] = 0
        row[-1] = 0
    return img_gray

### the hough algorithms part
def hough(edge_img):
    """
    Purpose:
        Main function of hough transform.
        Notice: Here I use the rho = x * sin(theta) + y * cos(theta). It is different from the hough equation.
        I do this to meet the convention of opencv so that to use opencv library easily later.
    Input:
        edge_img: binary image, the boundary image after edge detection.
    Output:
        res_dic: a dictionary, key is a tuple (rho, radian), value is the number of its key.
    """
    m, n = edge_img.shape
    res_dic = {}
    for i in range(m):
        for j in range(n):
            if edge_img[i,j] == 1:
                for degree in range(0,180):
                    radian = np.radians(degree)
                    rho = int(i * np.sin(radian) + j * np.cos(radian))
                    if ((rho, radian)) not in res_dic:
                        res_dic[(rho, radian)] = 0
                    res_dic[(rho, radian)] += 1
    return res_dic

def pick_by_count(lines, T):
    """
    Purpose:
        filter the lines which number of (rho, radian) greater than threshold T.
    Input:
        lines: a matrix, col_1: rho, col_2: radian, col_2: number of (rho, radian) tuple. It can be got by res_dic.
    Output:
        matrix, numpy array, each row represent a selected line. 
        
    """
    res = []
    for line in lines:
        if line[2] > T:
            res.append(line)  
    return np.asarray(res)

def pick_by_theta(lines, theta, gap):
    """
    Purpose:
        filter the lines by theta,
    Input:
        lines: a matrix, col_1: rho, col_2: radian, col_2: number of (rho, radian) tuple.
        theta: real, is a radian, will compared radians from col_2
        gap: the maximum difference between two theta.
    Output:
        matrix, numpy array, each row represent a selected line. 
    """
    res = []
    for line in lines:
        if abs(line[1] - theta) < gap:
            res.append(line)  
    return np.asarray(res)


def draw_lines(img, lines, loc="../task3_img/", name = "hough_res.jpg"):
    """
    Purpose: 
        draw lines in image
        not affect original imgage.
    """
    img = img.copy()
    for line in lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imwrite(loc+name, img)

def circle_hough(edge_img):
    res_dic = {}
    for x in range(edge_img.shape[0]):
        for y in range(edge_img.shape[1]):
            if edge_img[x,y] == 1:
                for r in range(18,28):
                    for t in range(0,360):
                        a = x - r * np.cos(t * np.pi / 180)
                        b = y - r * np.sin(t * np.pi / 180)
                        if (a,b,r) not in res_dic:
                            res_dic[(a,b,r)] = 0
                        res_dic[(a,b,r)] += 1
    return res_dic   

def draw_circles(img, circles, loc="../task3_img/", name="coin.jpg" ):
    img = img.copy()
    for cir in circles:
        cv2.circle(img,(cir[1], cir[0]), cir[2], (0,255,0), 1)
    cv2.imwrite(loc+name, img)



if __name__ == "__main__":
    img = cv2.imread('../task3_img/hough.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print("Start to do edge detection")
    ## get edges from four directions
    vertical = texture_filtering(img_gray, VERTICAL_SOBEL_3BY3)
    horizontal = texture_filtering(img_gray, HORIZONTAL_SOBEL_3BY3)
    pos45 = texture_filtering(img_gray, POS45_SOBEL_3BY3)
    neg45 = texture_filtering(img_gray, NEG45_SOBEL_3BY3)
    
    ## Find threshold through observation
    # vertical
    pixel_stat = count_pixels(np.abs(vertical))
    stat_list = np.asarray([[key,val] for key,val in pixel_stat.items() if key >100]).T
    plt.bar(stat_list[0],stat_list[1],align='center') # A bar chart
    plt.title("pixel numbers count (vert edge, value > 100)")
    plt.xlabel('pixel value')
    plt.ylabel('number')
    plt.savefig("../task3_img/vert_hist")
    plt.close()
    
    #horizontal
    pixel_stat = count_pixels(np.abs(horizontal))
    stat_list = np.asarray([[key,val] for key,val in pixel_stat.items() if key >100]).T
    plt.bar(stat_list[0],stat_list[1],align='center') # A bar chart
    plt.title("pixel numbers count (hori edge, value > 100)")
    plt.xlabel('pixel value')
    plt.ylabel('number')
    plt.savefig("../task3_img/hori_hist")
    plt.close()
    
    #pos45
    pixel_stat = count_pixels(np.abs(pos45))
    stat_list = np.asarray([[key,val] for key,val in pixel_stat.items() if key >100]).T
    plt.bar(stat_list[0],stat_list[1],align='center') # A bar chart
    plt.title("pixel numbers count (pos45 edge, value > 100)")
    plt.xlabel('pixel value')
    plt.ylabel('number')
    plt.savefig("../task3_img/pos45_hist")
    plt.close()
    
    #neg45
    pixel_stat = count_pixels(np.abs(neg45))
    stat_list = np.asarray([[key,val] for key,val in pixel_stat.items() if key >100]).T
    plt.bar(stat_list[0],stat_list[1],align='center') # A bar chart
    plt.title("pixel numbers count (neg45, value > 100)")
    plt.xlabel('pixel value')
    plt.ylabel('number')
    plt.savefig("../task3_img/neg45_hist")
    plt.close()
    
    ## Combined edges using threshold
    T = 50
    combined = combine_edge(horizontal,vertical,pos45,neg45,T)
    combined = ignore_border(combined)
    cv2.imwrite("../task3_img/combined.jpg", (combined*255).astype(np.uint8))
    
    ## preprocess binary image
    print("Start to preprocess binary image")
    #denoise
    struc_elem = np.ones((3,3)).astype(np.uint8)
    denoised = task1.denoising(method=2)(combined, struc_elem)
    cv2.imwrite("../task3_img/denoised.jpg", (denoised*255).astype(np.uint8))
    #closing
    struc_elem = np.ones((7,7)).astype(np.uint8)
    closed = task1.closing(denoised, struc_elem)
    cv2.imwrite("../task3_img/closed.jpg", (closed*255).astype(np.uint8))
    #extract boundary
    boundary = task1.boundary(closed)
    cv2.imwrite("../task3_img/boundary.jpg", (boundary*255).astype(np.uint8))
    
    ## Start using hough algorithms
    print("Start using hough algorithms")
    hres = hough(boundary)
    lines = np.asarray([[key[0], key[1], val] for key, val in hres.items()])
    #filter by theta and count
    #red lines
    T = 160
    theta_gap = 0.2
    theta_red = 3.1
    lines_red = pick_by_theta(lines, theta_red, theta_gap)
    res_red = pick_by_count(lines_red, T)
    draw_lines(img, res_red, loc="../task3_img/", name = "red_line.jpg")

    #blue lines
    T = 140
    theta_gap = 0.2
    theta_blue = 2.5
    lines_blue = pick_by_theta(lines, theta_blue, theta_gap)
    res_blue = pick_by_count(lines_blue, T)
    draw_lines(img, res_blue, loc="../task3_img/", name = "blue_lines.jpg")
    
    #Bonus Circle hough
    print("start hough circle")
    chres = circle_hough(boundary)
    circles = [item for item in chres.items() if chres[item[0]] >= 4]
    circles = np.asarray([[x[0][0], x[0][1], x[0][2], x[1]] for x in circles]).astype(int)
    draw_circles(img, circles)
    #Bonus using Canny edge detection
    print("hough circle on Canny edge")
    img = cv2.imread('../task3_img/hough.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    edges = (edges/255).astype(np.uint)
    chres = circle_hough(edges)
    circles = [item for item in chres.items() if chres[item[0]] >= 4]
    circles = np.asarray([[x[0][0], x[0][1], x[0][2], x[1]] for x in circles]).astype(int)
    draw_circles(img, circles, name="coin_edge.jpg")










