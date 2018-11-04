import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from task_1 import sift_match
UBIT = '50247057'
np.random.seed(sum([ord(c) for c in UBIT]))

"""
Cite: I learn a lot from opencv python tutorial for task2 part programming.
https://docs.opencv.org/3.4/d9/db7/tutorial_py_table_of_contents_calib3d.html
"""

def drawlines(img1,img2,lines,pts1,pts2,seed=30):
    ''' 
    cite: https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines 
    '''
    r,c,d = img1.shape
    rand = np.random.RandomState(seed)
    index = rand.permutation(len(lines))[0:10]
    for r,pt1,pt2 in np.asarray(list(zip(lines,pts1,pts2)))[index]:
        color = tuple(rand.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
    

if __name__ == "__main__":

	UBIT = '50247057'
	seed = sum([ord(c) for c in UBIT])

	img1 = cv2.imread('../task2_img/tsucuba_left.png')  # left image
	img2 = cv2.imread('../task2_img/tsucuba_right.png') # right image
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	##question 1
	kp1, des1, kp2, des2, good = sift_match(img1,img2,task="task2")

	## question 2
	pts2 = [kp2[m[0].trainIdx].pt for m in good]
	pts1 = [kp1[m[0].queryIdx].pt for m in good]
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
	# We select only inlier points
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]
	print("fundamental matrix:")
	print(F)

	##question 3
	# Find epilines corresponding to points in right image (second image) and
	# drawing its lines on left image
	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img5,img6 = drawlines(img1,img2,lines1,pts1,pts2, seed=seed)
	# Find epilines corresponding to points in left image (first image) and
	# drawing its lines on right image
	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3,img4 = drawlines(img2,img1,lines2,pts2,pts1,seed=seed)

	cv2.imwrite('../task2_img/task2_epi_left.jpg',img5)
	cv2.imwrite('../task2_img/task2_epi_right.jpg',img3)

	## question 4
	stereo = cv2.StereoBM_create(numDisparities=64, blockSize=25)
	disparity = stereo.compute(img1_gray,img2_gray)
	norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	cv2.imwrite('../task2_img/task2_disparity.jpg',(norm_image*255).astype(np.uint8))

