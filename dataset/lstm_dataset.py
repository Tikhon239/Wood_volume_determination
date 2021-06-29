import os
import re
import cv2
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, Subset

from torchvision import transforms

from typing import Sequence

from config import Config


class MySimpleDataset(Dataset):
    def __init__(self, path, data_name, number_of_strips=5):
        super().__init__()
        self.data_name = data_name
        self.number_of_strips = number_of_strips
        self.left_bound = 0
        self.right_bound = -1

        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.path_folders = self.get_path_folders(path)
        # берем только те папки, которые размечены
        self.path_folders = self.filter_path_folders(self.path_folders)

    def __len__(self):
        return len(self.path_folders)

    @staticmethod
    def get_path_folders(path):
        path_folders = []
        for x in sorted(os.listdir(path)):
            if x != ".DS_Store":
                for y in sorted(os.listdir(os.path.join(path, x))):
                    if y != ".DS_Store":
                        path_folders.append(os.path.join(path, x, y))
        return np.array(path_folders)

    @staticmethod
    def get_image_number(path):
        return int(re.sub('.*front(.*)\.jpg', '\g<1>', path))

    @staticmethod
    def get_end_of_path(path):
        return '/'.join(path.split('/')[-2:])

    def get_front_frames(self, path):
        end_of_path_folder = self.get_end_of_path(path)
        df = pd.read_csv(self.data_name)
        df_path_folder = np.array([self.get_end_of_path(df_path) for df_path in df.path.values])
        data_front_frames = df.front_frames[df_path_folder == end_of_path_folder]
        front_frames = [front_frame.strip("[]").replace("'", "").split(", ") for front_frame in data_front_frames]
        return [[self.get_image_number(frame) for frame in front_frame] for front_frame in front_frames]

    def filter_path_folders(self, path_folders, min_number_of_packs=1):
        true_path_folders = []
        for path_folder in path_folders:
            front_frames = self.get_front_frames(path_folder)
            if len(front_frames) >= min_number_of_packs:
                true_path_folders.append(path_folder)
        return np.array(true_path_folders)

    def get_path_front_images(self, path_folder):
        path_front_images = glob.glob(os.path.join(path_folder, "FrontJPG", "*jpg"))
        path_front_images.sort(key=lambda path: self.get_image_number(path))
        return path_front_images

    def get_image(self, image_path):
        front_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        h, w, c = front_image.shape
        if self.number_of_strips is not None:
            self.left_bound = (self.number_of_strips // 2) * w // self.number_of_strips
            self.right_bound = (1 + self.number_of_strips // 2) * w // self.number_of_strips

        front_image = torch.tensor(
            front_image[:, self.left_bound: self.right_bound].transpose(2, 0, 1),
            dtype=torch.float32
        )

        return self.transform(front_image)

    def get_images(self, path_folder):
        # считываем изображения
        path_front_images = self.get_path_front_images(path_folder)
        return torch.stack([self.get_image(image_path) for image_path in path_front_images], dim=0)

    def get_labels(self, path_folder):
        number_of_images = len(self.get_path_front_images(path_folder))
        labels = torch.zeros(number_of_images, dtype=torch.int8)

        front_frames = self.get_front_frames(path_folder)

        for front_frame in front_frames:
            labels[front_frame[:2]] = 1  # начало пачки
            labels[front_frame[-2:]] = 1  # конец пачки
            labels[front_frame[2:-2]] = 2  # сама пачка
        return labels

    def __getitem__(self, index):
        # можно добавить семплинг с сохранением порядка
        path_folder = self.path_folders[index]
        return self.get_images(path_folder), self.get_labels(path_folder)


class MyDataset(MySimpleDataset):
    def __init__(self, path, data_name, down_scale, config=Config(), number_of_strips=5):
        super().__init__(path, data_name, number_of_strips)

        self.down_scale = down_scale
        self.config = config

    def get_image(self, image_path):
        front_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        height, width, c = front_image.shape
        calibration_matrix = self.config.get_calibration_matrix(height, width)
        distortion_coefficients = self.config.get_distortion_coefficients()

        undistort_front_image = self.config.undistort_image(
            front_image, calibration_matrix, distortion_coefficients
        )

        undistort_scaled_front_image = cv2.resize(
            undistort_front_image, (0, 0), fx=1 / self.down_scale, fy=1 / self.down_scale
        )

        h, w, c = undistort_scaled_front_image.shape
        if self.number_of_strips is not None:
            self.left_bound = (self.number_of_strips // 2) * w // self.number_of_strips
            self.right_bound = (1 + self.number_of_strips // 2) * w // self.number_of_strips

        undistort_scaled_front_image = torch.tensor(
            undistort_scaled_front_image[:, self.left_bound: self.right_bound].transpose(2, 0, 1),
            dtype=torch.float32
        )

        return self.transform(undistort_scaled_front_image)


class MySubset(Subset):
    def __init__(self, dataset: MySimpleDataset, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)

        self.path_folders = dataset.path_folders[indices]
        self.get_front_frames = dataset.get_front_frames


def collate_fn(batch_samples):
    images, labels = list(zip(*batch_samples))

    seq_sizes = [image.shape[0] for image in images]

    image_batch = torch.cat(images, dim=0)
    label_batch = torch.full((len(seq_sizes), max(seq_sizes)), -1, dtype=torch.int8)
    for i, label in enumerate(labels):
        label_batch[i, :label.shape[0]] = label

    return image_batch, label_batch, seq_sizes


def balance_dataset_split(dataset,
                          test_size: float = 0.3,
                          random_state: int = 42):
    np.random.seed(random_state)
    dataset_size = len(dataset)
    train_indices, test_indices = train_test_split(
        np.arange(dataset_size),
        test_size=int(test_size * dataset_size),
        random_state=random_state
    )

    train_dataset = MySubset(dataset, indices=train_indices)
    test_dataset = MySubset(dataset, indices=test_indices)
    return train_dataset, test_dataset