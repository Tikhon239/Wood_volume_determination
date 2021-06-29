import numpy as np
import pandas as pd
import cv2
import re
import os
from glob import glob

from config import Config


def get_image_number(path):
    return int(re.sub('.*front(.*)\.jpg', '\g<1>', path))


def get_end_of_path(path):
    return '/'.join(path.split('/')[-2:])


def get_path_folders(path):
    path_folders = []
    for x in sorted(os.listdir(path)):
        if x != ".DS_Store":
            for y in sorted(os.listdir(os.path.join(path, x))):
                if y != ".DS_Store":
                    path_folders.append(os.path.join(path, x, y))
    return np.array(path_folders)


def get_path_front_images(path_folder):
    path_front_images = glob(os.path.join(path_folder, "FrontJPG", "*jpg"))
    path_front_images.sort(key=lambda path: get_image_number(path))
    return np.array(path_front_images)


def get_image(image_path, config=Config(), down_scale=1, undistort=True):
    front_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if not undistort:
        return cv2.resize(
            front_image, (0, 0), fx=1 / down_scale, fy=1 / down_scale
        )

    height, width, c = front_image.shape
    calibration_matrix = config.get_calibration_matrix(height, width)
    distortion_coefficients = config.get_distortion_coefficients()

    undistort_front_image = config.undistort_image(
        front_image, calibration_matrix, distortion_coefficients
    )

    return cv2.resize(
        undistort_front_image, (0, 0), fx=1 / down_scale, fy=1 / down_scale
    )


def get_front_frames(path, data_name):
    end_of_path_folder = get_end_of_path(path)
    df = pd.read_csv(data_name)
    df_path_folder = np.array([get_end_of_path(df_path) for df_path in df.path.values])
    data_front_frames = df.front_frames[df_path_folder == end_of_path_folder]
    front_frames = [front_frame.strip("[]").replace("'", "").split(", ") for front_frame in data_front_frames]
    return [[get_image_number(frame) for frame in front_frame] for front_frame in front_frames]