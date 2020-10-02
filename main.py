import os
from imutils import paths
import cv2
import utilities as util
import Combiner
import calibration
import matplotlib.pyplot as plt
import imutils


def main():

    #calib = calibration.calibration(visualtion=True)
    base_dataset_path = os.path.join(os.getcwd(), "datasets", "test_lab6")
    file_name = os.path.join(base_dataset_path, "imageData.txt")
    image_directory = base_dataset_path
    drone_location = os.path.join(base_dataset_path, "drone_postion.txt")
    write_img_dir_path = os.path.join(base_dataset_path, "results")

    all_images, data_matrix = util.importData(
        file_name, drone_location, image_directory)
    all_images = all_images[:12]
    data_matrix = data_matrix[:12]
    # for i in range(0,3):    
    #     all_images[i] = all_images[i][::10, ::10, :]
    #all_imgs_undistorted = calib.calibrate(all_images)
    # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    # (status, stitched) = stitcher.stitch(all_images)
    my_combiner = Combiner.Combiner(all_images, data_matrix)
    result = my_combiner.createMosaic()
    util.display("RESULT", result)
    if not os.path.exists(write_img_dir_path):
        os.makedirs(write_img_dir_path)
    cv2.imwrite(os.path.join(write_img_dir_path, "finalResult3.png"), result)


if __name__ == "__main__":
    main()
