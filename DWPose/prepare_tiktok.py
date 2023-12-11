import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from dwpose_utils import DWposeDetector


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def process(dwprocessor, input_image, detect_resolution):

    if not isinstance(dwprocessor, DWposeDetector):
        dwprocessor = DWposeDetector()

    with torch.no_grad():
        # input_image = HWC3(input_image)
        detected_map = dwprocessor(input_image)
        # detected_map = dwprocessor(resize_image(input_image, detect_resolution))
        # detected_map = HWC3(detected_map)

    return detected_map


dwprocessor = DWposeDetector()

# your dataset path
dataset_folder = '../../TikTok_dataset'
detect_resolution = 768
all_files = os.listdir(dataset_folder)

for folder in tqdm(all_files):
    folder_path = os.path.join(dataset_folder, folder)
    image_folder = os.path.join(folder_path, 'images')
    output_folder = os.path.join(folder_path, 'dwpose')

    if os.path.exists(output_folder):
        continue

    if os.path.exists(image_folder) and os.path.isdir(image_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for image_name in tqdm(os.listdir(image_folder),desc=f"process {folder}"):
            image_path = os.path.join(image_folder, image_name)
            output_path = os.path.join(output_folder, image_name)

            if os.path.isfile(image_path):
                input_image = cv2.imread(image_path)
                detected_map = process(dwprocessor, input_image, detect_resolution)
                cv2.imwrite(output_path, detected_map)