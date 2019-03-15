import utilities as util
import Combiner
import cv2
import os
import numpy as np


def main():

    base_dataset_path = os.path.join(os.getcwd(),"datasets/images")
    fileName = os.path.join(base_dataset_path,"imageData.txt")
    imageDirectory = base_dataset_path
    drone_location =os.path.join(base_dataset_path,"drone_postion.txt")
    write_img_dir_path = os.path.join(base_dataset_path,"results")

    allImages, dataMatrix = util.importData(fileName, drone_location, imageDirectory)
    myCombiner = Combiner.Combiner(allImages, dataMatrix)
    result = myCombiner.createMosaic()
    util.display("RESULT", result)
    if not os.path.exists(write_img_dir_path):
        os.makedirs(write_img_dir_path)
    cv2.imwrite(os.path.join(write_img_dir_path,"finalResult.png"), result)



if __name__ == "__main__":
    main()













