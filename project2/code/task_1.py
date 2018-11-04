import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

UBIT = '50247057'
np.random.seed(sum([ord(c) for c in UBIT]))


def sift_match(img_l,img_r,task="task1"):
    img_l_gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp_l,des_l = sift.detectAndCompute(img_l_gray,None)
    kp_r,des_r = sift.detectAndCompute(img_r_gray,None)
    
    cv2.imwrite("../"+task+"_img/"+task+"_sift1.jpg", cv2.drawKeypoints(img_l_gray, kp_l, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    cv2.imwrite("../"+task+"_img/"+task+"_sift2.jpg", cv2.drawKeypoints(img_r_gray, kp_r, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des_l,des_r,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img_res = cv2.drawMatchesKnn(img_l,kp_l,img_r,kp_r,good,None,flags=2)
    cv2.imwrite("../"+task+"_img/"+task+"_matches_knn.jpg",img_res)
    return kp_l, des_l, kp_r, des_r, good

def draw_match_img(img_l, img_r, kp_l, des_l, kp_r, des_r, good, seed):
    img_l_gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    ## Use the FLANN
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp_l[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_r[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h,w= img_l_gray.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img_r_gray = cv2.polylines(img_r_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    rand = np.random.RandomState(seed)
    index = rand.permutation(len(good))[0:10];

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = [matchesMask[i] for i in index], # draw only inliers
                   flags = 2)
    img5 = cv2.drawMatches(img_l_gray,kp_l,img_r_gray,kp_r, list(map(lambda x: x[0], [good[i] for i in index])) ,None,**draw_params)
    cv2.imwrite('../task1_img/task1_matches.jpg',img5)
    return M
def warpTwoImages(img1, img2, H):
    
    '''
    Cite: learn from 
    https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
    warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

if __name__ == "__main__":
    UBIT = '50247057'
    seed = sum([ord(c) for c in UBIT])
    filename_1 = '../task1_img/mountain1.jpg'
    filename_2 = '../task1_img/mountain2.jpg'
    img_l = cv2.imread(filename_1) # trainImage
    img_r = cv2.imread(filename_2) # queryImage
    img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    img_l_gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    kp_l, des_l, kp_r,des_r, good = sift_match(img_l, img_r, task="task1")
    M = draw_match_img(img_l, img_r, kp_l, des_l, kp_r, des_r, good, seed)
    print("homography matrix:")
    print(M)
    ## wrap left images
    res = warpTwoImages(img_r, img_l, M)
    cv2.imwrite('../task1_img/task1_pano.jpg',res)








