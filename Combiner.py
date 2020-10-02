import cv2
import numpy as np
import utilities as util
import geometry as gm
import copy
import time
import matplotlib.pyplot as plt


class Combiner:
    def __init__(self, imageList_, dataMatrix_):
        '''
        :param imageList_: List of all images in dataset.
        :param dataMatrix_: Matrix with all pose data in dataset.
        :return:
        '''
        self.imageList = []  # for storing the corrected image after projection transform.
        self.dataMatrix = (np.asarray(dataMatrix_)).astype(np.float)
        self.masks_backward = []
        self.masks_forward = []

        for i in range(0, len(imageList_)):
            # downsample the image to speed things up. 4000x3000 is huge!
            image = imageList_[i][::6, ::6, :]
            M = gm.computeUnRotMatrix(self.dataMatrix[i, :])

            # TODO create list of masks for feature detection for each image.
            mask_backward, mask_forward = create_mask(
                self.dataMatrix, i, image.shape, full_mask=True)
            
            

            # Perform a perspective transformation based on pose information.
            # Ideally, this will make each image look as if it's viewed from the top.
            # We assume the ground plane is perfectly flat.
            correctedImage = gm.warpPerspectiveWithPadding(image, M)
            mask_backward = gm.warpPerspectiveWithPadding(mask_backward, M)
            mask_forward = gm.warpPerspectiveWithPadding(mask_forward, M)
            self.masks_backward.append(mask_backward)
            self.masks_forward.append(mask_forward)
            # store only corrected images to use in combination
            self.imageList.append(correctedImage)
        self.resultImage = self.imageList[0]

    def createMosaic(self):
        start_time = time.time()  # start counting time
        for i in range(1, len(self.imageList)):
            print("elapsed time: {0}".format(time.time()-start_time))
            print("image number: {0} \n".format(i))
            self.combine(i)
        return self.resultImage

    def combine(self, index2):
        '''
        :param index2: index of self.imageList and self.kpList to combine with self.referenceImage and self.referenceKeypoints
        :return: combination of reference image and image at index 2
        '''

        # Attempt to combine one pair of images at each step. Assume the order in which the images are given is the best order.
        # This intorduces drift!
        image1 = copy.copy(self.imageList[index2 - 1])
        image2 = copy.copy(self.imageList[index2])

        '''
        Descriptor computation and matching.
        Idea: Align the images by aligning features.
        '''
        detector = cv2.xfeatures2d.SIFT_create()

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        ret1, mask1 = cv2.threshold(
            gray1, 1, 255, cv2.THRESH_BINARY)  # creating a mask
        kp1, descriptors1 = detector.detectAndCompute(
            gray1, (self.masks_forward[index2-1]).astype(np.uint8))  # kp = keypoints

        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        ret2, mask2 = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
        kp2, descriptors2 = detector.detectAndCompute(
            gray2, (self.masks_backward[index2]).astype(np.uint8))

        print("kp1={0}, kp2={1} finding matches.. \n".format(
            len(kp1), len(kp2)))

        # maching features
        FLANN_INDEX_KDTREE = 0
        MIN_MATCH_COUNT = 10
        Visualize_matches = False  # set to True to view matches

        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        #bf = cv2.BFMatcher()

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        #matches = bf.knnMatch(descriptors1,descriptors2, k=2)

        print("{0} matches found \n".format(len(matches)))

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        '''
        Compute Affine Transform
        Idea: Because we corrected for camera orientation, an affine transformation *should* be enough to align the images
        '''
        print(len(good))
        if len(good) > MIN_MATCH_COUNT:

            # NumPy syntax for extracting location data from match data structure in matrix form
            src_pts = np.float32(
                [kp1[gmatch.queryIdx].pt for gmatch in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[gmatch.trainIdx].pt for gmatch in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            matchesMask = mask.ravel().tolist()

            h, w = gray1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                              [w-1, 0]]).reshape(-1, 1, 2)

            if Visualize_matches == True:
                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)
                matchDrawing = cv2.drawMatches(
                    gray1, kp1, gray2, kp2, good, None, **draw_params)
                plt.imshow(matchDrawing, 'gray'), plt.show()

        else:
            print("Not enough matches are found - %d/%d" %
                  (len(good), MIN_MATCH_COUNT))
            matchesMask = None

        '''
        Compute 4 Image Corners Locations
        Idea: Same process as warpPerspectiveWithPadding() except we have to consider the sizes of two images. Might be cleaner as a function.
        '''

        height1, width1 = image1.shape[:2]
        height2, width2 = image2.shape[:2]



        corners1 = np.float32(
            ([0, 0], [0, height1], [width1, height1], [width1, 0]))
        corners2 = np.float32(
            ([0, 0], [0, height2], [width2, height2], [width2, 0]))
        warpedCorners2 = np.zeros((4, 2))
        for i in range(0, 4):
            cornerX = corners2[i, 0]
            cornerY = corners2[i, 1]
            warpedCorners2[i, 0] = (H[0, 0]*cornerX + H[0, 1]*cornerY +
                                    H[0, 2])/(H[2, 0]*cornerX + H[2, 1]*cornerY + H[2, 2])
            warpedCorners2[i, 1] = (H[1, 0]*cornerX + H[1, 1]*cornerY +
                                    H[1, 2])/(H[2, 0]*cornerX + H[2, 1]*cornerY + H[2, 2])
        allCorners = np.concatenate((corners1, warpedCorners2), axis=0)
        [yMin, xMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5 -10)
        [yMax, xMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5 + 10)
        '''Compute Image Alignment and Keypoint Alignment'''
        translation = np.float32(([1, 0, -1*xMin], [0, 1, -1*yMin], [0, 0, 1]))
        # images must be translated to be 100% visible in new canvas
        fullTransformation = np.dot(translation, H)
        warpedResImg = cv2.warpPerspective(
            self.resultImage, fullTransformation, (xMax-xMin, yMax-yMin))

        warpedImage2 = cv2.warpPerspective(
            image2, translation, (xMax-xMin, yMax-yMin))
        self.masks_backward[index2] = cv2.warpPerspective(
            self.masks_backward[index2], translation, (xMax-xMin, yMax-yMin))
        self.masks_forward[index2] = cv2.warpPerspective(
            self.masks_forward[index2], translation, (xMax-xMin, yMax-yMin))

        # crucial: update old images for future feature extractions
        self.imageList[index2] = copy.copy(warpedImage2)
        resGray = cv2.cvtColor(self.resultImage, cv2.COLOR_BGR2GRAY)
        warpedResGray = cv2.warpPerspective(
            resGray, fullTransformation, (xMax-xMin, yMax-yMin))

        '''Compute Mask for Image Combination'''
        ret, mask1 = cv2.threshold(
            warpedResGray, 1, 255, cv2.THRESH_BINARY_INV)
        mask3 = np.float32(mask1)/255

        # add margin to mask to avoid black margin
        mask3 = add_margin_to_mask(mask3)

        # apply mask
        warpedImage2[:, :, 0] = warpedImage2[:, :, 0]*mask3
        warpedImage2[:, :, 1] = warpedImage2[:, :, 1]*mask3
        warpedImage2[:, :, 2] = warpedImage2[:, :, 2]*mask3
        result = merge_images(warpedResImg,warpedImage2)        
        # result = warpedResImg + warpedImage2
        result = cv2.addWeighted(warpedResImg, 1, warpedImage2, 1, 0.0)
        # visualize and save result
        self.resultImage = result

        return result


    def stitchMatches(self,image1,image2,homography):
        #gather x and y axis of images that will be stitched
        height1, width1 = image1.shape[0], image1.shape[1]
        height2, width2 = image2.shape[0], image2.shape[1]
        #create blank image that will be large enough to hold stitched image
        blank_image = np.zeros(((width1 + width2),(height1 + height2),3),np.uint8)
        #stitch image two into the resulting image while using blank_image 
        #to create a large enough frame for images
        result = cv2.warpPerspective((image1),homography,blank_image.shape[0:2])
        #numpy notation for slicing a matrix together allows you to see the image
        result[0:image2.shape[0], 0:image2.shape[1]] = image2
        return result

def create_mask(matrix_data, index, mask_size, full_mask = False):
    delta_movment = 1
    mask_forward = np.zeros(mask_size[:2])
    mask_backward = np.zeros(mask_size[:2])
    X, Y, Z, Yow, Pit, Rol = matrix_data[index]
    X_forward, Y_forward, Z_forward = 0, 0, 0
    X_back, Y_back, Z_back = 0, 0, 0

    if index > 0:
        X_back, Y_back, Z_back, Yow, Pit, Rol = matrix_data[index-1]
    if index < len(matrix_data):
        X_forward, Y_forward, Z_forward, Yow, Pit, Rol = matrix_data[index-1]

    direction = 1  # 1 = down-up,2=down-right,3=down-left,  4=up-right,5=up-left,6=up-down,   7=left-up, 8=left-right, 9=left-down,   10=right-up,11=right-left,12right-down
    # TODO switch direcion and create 2 masks.

    if direction == 1:
        mask_forward[(int(mask_size[0]/3)):int((mask_size[0]/3)*2),
                    (int(mask_size[1]/3)):int((mask_size[1]/3)*2)] = 255
        mask_backward[(int(mask_size[0]/3)):int((mask_size[0]/3)*2),
                    (int(mask_size[1]/3)):int((mask_size[1]/3)*2)] = 255
        mask_forward[:(int(mask_size[0]/2)), :] = 255
        mask_backward[int(mask_size[0]/2):, :] = 255
    
    if full_mask or direction == 13:
        mask_backward[:, :] = 255
        mask_forward[:, :] = 255

    return mask_backward, mask_forward


# def add_margin_to_mask(img_mask, margin=2):
#     new_img = np.ones(img_mask.shape)
#     for ix in range(img_mask.shape[0]):
#         for iy in range(img_mask.shape[1]):
#             if img_mask[ix, iy] == 0:
#                 new_img[ix, iy] = 0
#                 for i in range(0, margin):
#                     if ix-i >= 0:
#                         new_img[ix-i, iy] = 0
#                     if ix+i < img_mask.shape[0]:
#                         new_img[ix+i, iy] = 0
#                     if iy-i >= 0:
#                         new_img[ix, iy-i] = 0
#                     if iy+i < img_mask.shape[1]:
#                         new_img[ix, iy+i] = 0
#     return new_img
                
def add_margin_to_mask(img_mask, margin=1):
    new_img = copy.copy(img_mask)
    for ix in range(img_mask.shape[0]):
        for iy in range(img_mask.shape[1]):
            if img_mask[ix, iy] == 0:
                if ix-1 >= 0:
                    if img_mask[ix-1, iy] == 1:
                        for i in range(0, margin):
                            new_img[ix+i, iy] = 1 

                if iy-1 >= 0:
                    if img_mask[ix, iy-1] == 1:
                        for i in range(0, margin):
                            new_img[ix, iy+i] = 1 

                if ix+1 < img_mask.shape[0]:
                    if img_mask[ix+1, iy] == 1:
                        for i in range(0, margin):
                            new_img[ix-i, iy] = 1 

                if iy+1 < img_mask.shape[1]:
                    if img_mask[ix, iy+1] == 1:
                        for i in range(0, margin):
                            new_img[ix, iy+1] = 1
    return new_img
                
def merge_images(im1, im2):
    new_image = np.zeros(im1.shape)
    for i in range(im1.shape[0]):
        for j in range(im2.shape[1]): 
            if np.array_equal(im1[i,j], [0,0,0]):
                new_image[i,j] == im2[i,j]
            elif np.array_equal(im2[i,j], [0,0,0]):
                new_image == im1[i,j]
            if (np.array_equal(im2[i,j], [0,0,0]))==False and (np.array_equal(im1[i,j], [0,0,0]))==False:
                new_image[i,j] = np.round(im1[i,j] + im2[i,j] / 2)
    return new_image