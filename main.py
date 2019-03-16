import os
import cv2
import utilities as util
import Combiner


def main():
    base_dataset_path = os.path.join(os.getcwd(), "datasets", "images")
    file_name = os.path.join(base_dataset_path, "imageData.txt")
    image_directory = base_dataset_path
    drone_location = os.path.join(base_dataset_path, "drone_postion.txt")
    write_img_dir_path = os.path.join(base_dataset_path, "results")

    all_images, data_matrix = util.importData(file_name, drone_location, image_directory)
    all_images = all_images[:12]
    data_matrix = data_matrix[:12]
    my_combiner = Combiner.Combiner(all_images, data_matrix)
    result = my_combiner.createMosaic()
    util.display("RESULT", result)
    if not os.path.exists(write_img_dir_path):
        os.makedirs(write_img_dir_path)
    cv2.imwrite(os.path.join(write_img_dir_path, "finalResult3.png"), result)


if __name__ == "__main__":
    main()
