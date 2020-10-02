import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# https://stackoverflow.com/questions/22656698/perspective-correction-in-opencv-using-python

# img = cv.imread('/home/hezi/final_project/final_project-master/datasets/images/P7100016.JPG',1) # trainImage
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# w,h,d = img.shape
# src = np.float32([(0,     0),
#                   (h,  0),
#                   (h,    w),
#                   (0,  w)])

# dst = np.float32([(0,     0),
#                   (h,  0),
#                   (h,    w),
#                   (0,  w)])

# Matt = cv.getPerspectiveTransform(src, dst)
# warped = cv.warpPerspective(img, Matt, (w, h), flags=cv.INTER_LINEAR)
# plt.imshow(warped); plt.show()
# pass


# def unwarp(img, src, dst, testing):
#     h, w = img.shape[:2]
#     # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
#     M = cv.getPerspectiveTransform(src, dst)
#     # use cv2.warpPerspective() to warp your image to a top-down view
#     warped = cv.warpPerspective(img, M, (w, h), flags=cv.INTER_LINEAR)

#     if testing:
#         f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#         f.subplots_adjust(hspace=.2, wspace=.05)
#         ax1.imshow(img)
#         x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
#         y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
#         ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
#         ax1.set_ylim([h, 0])
#         ax1.set_xlim([0, w])
#         ax1.set_title('Original Image', fontsize=30)
#         ax2.imshow(cv.flip(warped, 1))
#         ax2.set_title('Unwarped Image', fontsize=30)
#         plt.show()
#     else:
#         return warped, M


# im = cv.imread('/home/hezi/final_project/final_project-master/datasets/images/P7100028.JPG',1) # trainImage

# w, h = im.shape[0], im.shape[1]
# # We will first manually select the source points 
# # we will select the destination point which will map the source points in
# # original image to destination points in unwarped image
# src = np.float32([(0, 0),
#                   (3000, 3000),
#                   (h, 0),
#                   (h, w)])

# dst = np.float32([(0, 0),
#                   (0, w),
#                   (h, 0),
#                   (h, w)])

# unwarp(im, src, dst, True)



import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img1 = cv.imread('box.png',0)          # queryImage
img2 = cv.imread('box_in_scene.png',0) # trainImage
# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()