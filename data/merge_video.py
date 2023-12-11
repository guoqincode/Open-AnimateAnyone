import cv2
import os
from tqdm import tqdm

dataset_folder = '../../TikTok_dataset'
fps = 30 # tiktok dataset fps=30

all_files = os.listdir(dataset_folder)



for folder in tqdm(all_files):
    folder_path = os.path.join(dataset_folder, folder)
    # image_folder = os.path.join(folder_path, 'images')
    image_folder = os.path.join(folder_path, 'dwpose')
    # video_path = os.path.join(folder_path, f"{folder}.mp4")
    video_path = os.path.join(folder_path, f"{folder}_dwpose.mp4")

    if os.path.exists(video_path):
        continue

    if os.path.exists(image_folder) and os.path.isdir(image_folder):
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        size = (width, height)

        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        for image in images:
            img_path = os.path.join(image_folder, image)
            img = cv2.imread(img_path)
            video.write(img)

        video.release()
