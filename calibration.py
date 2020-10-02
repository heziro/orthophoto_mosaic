import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt


class calibration:

    def __init__(self, visualtion=False):
        
        self.fixed_image_list = []
        self.visualtion = visualtion
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS +
                    cv.TERM_CRITERIA_MAX_ITER, 27, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        images = glob.glob('/home/hezi/Desktop/calibrate2/*.jpg')
        for fname in images:
            img = cv.imread(fname)
            self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(self.gray, (9, 6), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(objp)
                corners2 = cv.cornerSubPix(
                    self.gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (9, 6), corners2, ret)
                if visualtion == True:
                    im = cv.resize(img, (1296, 972))
                    cv.imshow('img', im)
                    cv.waitKey(500)
        if visualtion == True:
            cv.destroyAllWindows()

    def calibrate(self, imgs_to_calibrate):
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
            self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)

        # Undistortion imgs

        # TODO load all images.
        for img in imgs_to_calibrate:
            h, w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), 1, (w, h))

            # undistort
            dst = cv.undistort(img, self.mtx, self.dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            self.error_estimation() # calc error 
            self.fixed_image_list.append(dst)
        return self.fixed_image_list



    def error_estimation(self):
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(self.objpoints)) )