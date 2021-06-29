import cv2
import numpy as np

import torch

from dataclasses import dataclass


@dataclass
class Config:
    f: float = 10
    cx: float = 0
    cy: float = 0

    k1: float = -2e-5
    k2: float = 0
    p1: float = 0
    p2: float = 0
    k3: float = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_distortion_coefficients(self):
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    def get_calibration_matrix(self, height, width):
        return np.array([
            [self.f, 0, self.cx + 0.5 * width],
            [0, self.f, self.cy + 0.5 * height],
            [0, 0, 1]
        ])

    @staticmethod
    def undistort_image(image, calibration_matrix, distortion_coefficients):
        undistort_image = cv2.undistort(image, calibration_matrix, distortion_coefficients, None)
        return undistort_image