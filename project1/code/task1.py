import numpy as np
import cv2
import math
import mynumpy as mnp

VERTICAL_SOBEL_3BY3 = np.array([[1,0,-1],
                          [2,0,-2],
                          [1,0,-1]])

HORIZONTAL_SOBEL_3BY3 = np.array([[1,2,1],
                         [0,0,0],
                         [-1,-2,-1]])


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
    
    img_gray = np.ndarray.tolist(img_gray)
    img_gray = np.asarray(mnp.pad(img_gray, row_pad, row_pad, col_pad, col_pad, 0))
    img_res = np.asarray(mnp.zeros(img_gray.shape[0], img_gray.shape[1]))
    
    flipped_kernel = np.asarray((mnp.flip(np.ndarray.tolist(kernel))))
    for i in range(row_pad, img_gray.shape[0] - row_pad):
        for j in range(col_pad, img_gray.shape[1] - col_pad):
            patch = mnp.inner_product(img_gray[i-row_pad:i+row_pad+1, j-col_pad:j+col_pad+1], flipped_kernel)
            img_res[i,j] = mnp.sum_all(patch)
    return img_res[row_pad: img_res.shape[0] - row_pad, col_pad:img_res.shape[1] - col_pad]


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
        return (img - mnp.min_all(img)) / (mnp.max_all(img) - mnp.min_all(img))
    elif method == 2:
        return np.asarray(mnp.abs_all(img)) / mnp.max_all(mnp.abs_all(img))
    else :
        print("method is 1 or 2")


# magnitude of edges (conbining horizontal and vertical edges)
def magnitude_edges(edge_x, edge_y):
    """
    Purpose: 
        Combine the vertical image and horizontal image.
    Input:
        edge_x: two dimension matrix
            the image filted by VERTICAL_SOBEL
        edge_y: two dimension matrix
            the image filted by HORIZONTAL_SOBEL
    Output:
        edge_magnitude: two dimension matrix
            the image combined by edge_x and edge_y
        
    """
    edge_magnitude = np.sqrt(edge_x ** 2 + edge_y ** 2)
    edge_magnitude /= mnp.max_all(edge_magnitude)
    return edge_magnitude


def direction_edge(edge_x, edge_y):
    edge_direction = np.arctan(edge_y / (edge_x + 1e-3)) * 180. / np.pi
    edge_direction /= mnp.max_all(edge_direction)
    return edge_direction


if __name__ == "__main__":
	img = cv2.imread("../task1_img/task1.png", 0)
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	name = "../task1_img" + "/image" + ".png"
	cv2.imwrite(name, img)

	# Computing vertical edges
	edge_x = texture_filtering(img,VERTICAL_SOBEL_3BY3)
	cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
	cv2.imshow('edge_x_dir', edge_x)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	name = "../task1_img" + "/edge_x" + ".png"
	cv2.imwrite(name, edge_x)

	# Eliminate zero values with method 1
	pos_edge_x_1 = eliminate_zero(edge_x, 1)
	cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
	cv2.imshow('pos_edge_x_dir', pos_edge_x_1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	name = "../task1_img" + "/pos_edge_x_1" + ".png"
	cv2.imwrite(name, pos_edge_x_1 * 255)

	# Eliminate zero values with method 2
	pos_edge_x_2 = eliminate_zero(edge_x, 2)
	cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
	cv2.imshow('pos_edge_x_dir', pos_edge_x_2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	name = "../task1_img" + "/pos_edge_x_2" + ".png"
	cv2.imwrite(name, pos_edge_x_2 * 255)

	# Computing horizontal edges
	edge_y = texture_filtering(img,HORIZONTAL_SOBEL_3BY3)
	cv2.namedWindow('edge_y_dir', cv2.WINDOW_NORMAL)
	cv2.imshow('edge_y_dir', edge_y)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	name = "../task1_img" + "/edge_y" + ".png"
	cv2.imwrite(name, edge_y)

	# Eliminate zero values with method 1
	pos_edge_y_1 = eliminate_zero(edge_y, 1)
	cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
	cv2.imshow('pos_edge_y_dir', pos_edge_y_1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	name = "../task1_img" + "/pos_edge_y_1" + ".png"
	cv2.imwrite(name, pos_edge_y_1 * 255)

	# Eliminate zero values with method 2
	pos_edge_y_2 = eliminate_zero(edge_y, 2)
	cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
	cv2.imshow('pos_edge_y_dir', pos_edge_y_2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	name = "../task1_img" + "/pos_edge_y_2" + ".png"
	cv2.imwrite(name, pos_edge_y_2 * 255)

	# magnitude of edges (conbining horizontal and vertical edges)
	edge_magnitude = magnitude_edges(edge_x, edge_y)
	cv2.namedWindow('edge_magnitude', cv2.WINDOW_NORMAL)
	cv2.imshow('edge_magnitude', edge_magnitude)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	name = "../task1_img" + "/edge_magnitude" + ".png"
	cv2.imwrite(name, edge_magnitude * 255)

	edge_direction = direction_edge(edge_x, edge_y)
	cv2.namedWindow('edge_direction', cv2.WINDOW_NORMAL)
	cv2.imshow('edge_direction', edge_direction)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	name = "../task1_img" + "/edge_direction" + ".png"
	cv2.imwrite(name, edge_direction * 255)


	print("Original image size: {:4d} x {:4d}".format(img.shape[0], img.shape[1]))
	print("Resulting image size: {:4d} x {:4d}".format(edge_magnitude.shape[0], edge_magnitude.shape[1]))
















