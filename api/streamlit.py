import streamlit as st
import numpy as np
import cv2
import re
import os
from glob import glob
import matplotlib.pyplot as plt
import joblib


from dataclasses import dataclass


@dataclass
class Config:
    f  : float = 10
    cx : float = 0
    cy : float = 0

    k1 : float = -2e-5
    k2 : float = 0
    p1 : float = 0
    p2 : float = 0
    k3 : float = 0
    
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

def get_image_number(path):
    return int(re.sub('.*front(.*)\.jpg', '\g<1>', path))

def get_path_folders(path):
    path_folders = []
    for x in sorted(os.listdir(path)):
        if x != ".DS_Store":
            for y in sorted(os.listdir(os.path.join(path, x))):
                if y != ".DS_Store":
                    path_folders.append(os.path.join(path, x, y))
    return path_folders

def get_image(image_path, config = Config(), down_scale = 1):
    front_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    height, width, c = front_image.shape
    calibration_matrix = config.get_calibration_matrix(height, width)
    distortion_coefficients = config.get_distortion_coefficients()
        
    undistort_front_image = config.undistort_image(
        front_image, calibration_matrix, distortion_coefficients
    )

    return cv2.resize(
        undistort_front_image, (0, 0), fx = 1 / down_scale, fy = 1 / down_scale
    )

def get_path_front_images(path_folder):
        path_front_images = glob(os.path.join(path_folder, "FrontJPG", "*jpg"))
        path_front_images.sort(key=lambda path : get_image_number(path))
        return path_front_images

def get_gabor_filter(size = 15, scale = 9, angle = np.pi/2):
    sigma = 1.5 * scale / 6
    
    kern = cv2.getGaborKernel((size, size), sigma, angle, scale, gamma = 1, psi = np.pi) 
    kern[kern>0] /= kern[kern>0].sum()
    kern[kern<0] /= np.abs(kern[kern<0].sum())
    return kern

def bfs_explore(start_y, start_x, image, color_map, color, binary_theshold):
    h, w = image.shape[:2]
    queue = [[start_y, start_x]]
    while len(queue) > 0:
        y, x = queue.pop()
        color_map[y, x] = color
        for dy in range(max(0, y - 1), min(y + 1, h - 1) + 1):
            for dx in range(max(0, x - 1), min(x + 1, w - 1) + 1):
                if color_map[dy, dx] == 0 and image[dy, dx] > binary_theshold:
                    queue.append([dy, dx])
                
def bfs(image, binary_theshold = 0):
    h, w = image.shape[:2]
    color_map = np.zeros((h, w), dtype = np.uint8)
    cur_color = 1
    y_coords, x_coords = np.where(image > binary_theshold)
    for (y, x) in zip(y_coords, x_coords):
        if color_map[y, x] == 0:
            bfs_explore(y, x, image, color_map, cur_color, binary_theshold)
            cur_color += 1
    return color_map

def search_lines(image, number_of_strips = 5, threshold = 10):
    # threshold - порог бинаризации карты градиентов
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = image.shape[:2]

    # банк фильтров Габора
    size = 15
    scale = 9
    angle = np.pi/2
    kern = get_gabor_filter(size = size, scale = scale, angle = angle)
    
    # вырезаем полоску
    left_bound = (number_of_strips // 2) * w // number_of_strips
    right_bound = (1 + number_of_strips // 2) * w // number_of_strips
    res_gray_image = gray_image[:, left_bound : right_bound]
    nh, nw = res_gray_image.shape[:2]

    # применение банка фильров и выбор максимума
    res_grad = cv2.filter2D(res_gray_image, -1, kern)
    
    res_morphology = (res_grad > threshold).astype('uint8')
    # применение морфологии
    # убираем шум
    open_kernel = np.ones((2, 10))
    res_morphology = cv2.morphologyEx(res_morphology, cv2.MORPH_OPEN, open_kernel)
    # боремся с рейками
    close_kernel = np.ones((2, 30))
    res_morphology = cv2.morphologyEx(res_morphology, cv2.MORPH_CLOSE, close_kernel)

    # поиск прямых при помощи поиска в ширину
    bfs_color = np.zeros((nh, nw, 3), dtype = np.uint8)
    bfs_res = bfs(res_morphology, 0)
    bfs_cout = 0
    bfs_cout_long_lines = 0
    for color in range(1, bfs_res.max() + 1):
        y_coords, x_coords = np.where(bfs_res == color)
        x1, x2 = x_coords.min(), x_coords.max()
        if (x2 - x1) > 0.33 * nw and (x1 < 0.1 * nw or x2 > 0.9 * nw):
            bfs_cout += 1
            bfs_cout_long_lines += (x2 - x1) > 0.75 * nw
            y1 = y2 = int(y_coords.mean())
            cv2.line(bfs_color, (x1, y1), (x2, y2), np.random.uniform(0, 255, 3), 2)
    
    # визуализация результатов
    alpha = 0.8
    mask = np.zeros_like(image)
    mask[:, left_bound : right_bound] = [255, 0, 0]
    mixed_image = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0)
    res_morphology_rgb = cv2.cvtColor(255 * res_morphology, cv2.COLOR_GRAY2RGB)
    res_image = np.hstack((np.hstack((mixed_image, res_morphology_rgb)), bfs_color))
    return bfs_cout, bfs_cout_long_lines, res_image


def classify_image(image_path, config = Config(), down_scale = 1, number_of_strips = 5, threshold = 7):
    image = get_image(image_path, config, down_scale)
    bfs_cout, bfs_cout_long_lines, _ = search_lines(image, number_of_strips = number_of_strips)
    return bfs_cout_long_lines > threshold and bfs_cout_long_lines/bfs_cout > 0.5

def mapping(images_path, config = Config(), down_scale = 1, number_of_strips = 5, threshold = 7):
    classify_image_lambda = lambda image_path : classify_image(image_path,
                                                               config = config, 
                                                               down_scale = down_scale, 
                                                               number_of_strips = number_of_strips, 
                                                               threshold = threshold
                                                               )
    process = joblib.delayed(classify_image_lambda)
    mapping_result = joblib.Parallel(n_jobs=4)(process(image_path) for image_path in images_path)
    return np.array(mapping_result, dtype = np.int8)

@st.cache
def mapping(images_path, config = Config(), down_scale = 1, number_of_strips = 5, threshold = 7):
    classify_image_lambda = lambda image_path : classify_image(image_path,
                                                               config = config, 
                                                               down_scale = down_scale, 
                                                               number_of_strips = number_of_strips, 
                                                               threshold = threshold
                                                               )
    mapping_result = np.zeros(len(images_path), dtype = np.int)
    for i, image_path in enumerate(images_path):
        mapping_result[i] = classify_image_lambda(image_path)
    return mapping_result


root_path = 'data'
config = Config()

path_folder = st.selectbox('Путь папки:', get_path_folders(root_path))
path_front_images = get_path_front_images(path_folder)

down_scale = 4#st.slider('Down scale', 1, 10, 4)
# бьем изображение на number_of_strips полосок и выбираем среднюю  
number_of_strips = 5#st.slider('На сколько полосок бьем фотографию', 1, 7, 5, 2)

st.header('Количество обнаруженных прямых')
front_image_number = st.slider('Номер фотографии:', 0, len(path_front_images) - 1)
path_front_image = path_front_images[front_image_number]
selected_image = get_image(path_front_image, config, down_scale)
bfs_cout, bfs_cout_long_lines, res_image = search_lines(selected_image, number_of_strips = number_of_strips)
# количество обнаруженных прямых
st.write('Общее количество:', bfs_cout)
st.write('Длинные прямые:', bfs_cout_long_lines)
st.image(res_image, output_format="PNG")

try:
    mapping_result = np.load(os.path.join(path_folder, "mapping_result.npy"))
except:
    mapping_result = mapping(path_front_images,
        config = config,
        down_scale = down_scale,
        number_of_strips = number_of_strips,
        threshold = 7
    )

fig, ax = plt.subplots(figsize=(10,2))
ax.plot(mapping_result, "o-")
ax.plot(front_image_number, mapping_result[front_image_number], 'ro')
st.pyplot(fig)