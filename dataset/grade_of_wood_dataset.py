import os
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, Subset

from torchvision import transforms

from typing import Sequence

from config import Config


class MySimpleDataset(Dataset):
    def __init__(self, path, data_name):
        super().__init__()

        self.image_paths = []
        self.image_labels = []
        self.label_encoder = LabelEncoder()
        self.get_path_images_and_labels(path, data_name)

        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_labels)

    def get_path_images_and_labels(self, path, data_name):
        df = pd.read_csv(data_name)
        df_path_folder = np.array(['/'.join(df_path.split('/')[-2:]) for df_path in df.path.values])

        self.label_encoder.fit(df.sort)
        for x in sorted(os.listdir(path)):
            if x != ".DS_Store":
                for y in sorted(os.listdir(os.path.join(path, x))):
                    if y != ".DS_Store":
                        end_of_path_folder = os.path.join(x, y)
                        mask = df_path_folder == end_of_path_folder
                        if mask.sum() > 0:
                            front_labels = self.label_encoder.transform(df.sort[mask])
                            front_paths = [[os.path.join(path, x, y, "FrontJPG", img_name) for img_name in
                                            front_frame.strip("[]").replace("'", "").split(", ")] for front_frame in
                                           df.front_frames[mask]]
                            front_paths_size = list(map(len, front_paths))

                            self.image_paths.extend(np.hstack(front_paths))
                            self.image_labels.extend(np.repeat(front_labels, front_paths_size))

        self.image_paths = np.array(self.image_paths)
        self.image_labels = np.array(self.image_labels)

    def get_image(self, image_path):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return self.transform(torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_label = self.image_labels[index]

        return self.get_image(image_path), image_label


class MyDataset(MySimpleDataset):
    def __init__(self, path, down_scale, data_name, config=Config()):
        super().__init__(path, data_name)

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

        return self.transform(torch.tensor(undistort_scaled_front_image.transpose(2, 0, 1), dtype=torch.float32))


class MySubset(Subset):
    def __init__(self, dataset: MySimpleDataset, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)

        self.image_paths = dataset.image_paths[indices]
        self.image_labels = dataset.image_labels[indices]


def balance_dataset_split(dataset,
                          test_size: float = 0.3,
                          random_state: int = 42):
    np.random.seed(random_state)
    dataset_size = len(dataset)
    targets = dataset.image_labels
    train_indices, test_indices = train_test_split(
        np.arange(dataset_size),
        test_size=int(test_size * dataset_size),
        stratify=targets,
        random_state=random_state
    )
    train_dataset = MySubset(dataset, indices=train_indices)
    test_dataset = MySubset(dataset, indices=test_indices)
    return train_dataset, test_dataset