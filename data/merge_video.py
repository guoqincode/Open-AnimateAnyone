import cv2
import os
from tqdm import tqdm

# 定义数据集文件夹路径和视频的FPS
dataset_folder = '../../TikTok_dataset'
fps = 30

all_files = os.listdir(dataset_folder)

# all_files = all_files[:120]
# all_files = all_files[140:230]
# all_files = all_files[247:]

# 遍历数据集文件夹中的所有子文件夹
for folder in tqdm(all_files):
    folder_path = os.path.join(dataset_folder, folder)
    # image_folder = os.path.join(folder_path, 'images')
    image_folder = os.path.join(folder_path, 'dwpose')
    # video_path = os.path.join(folder_path, f"{folder}.mp4")
    video_path = os.path.join(folder_path, f"{folder}_dwpose.mp4")

    if os.path.exists(video_path):
        continue

    # 如果image文件夹存在
    if os.path.exists(image_folder) and os.path.isdir(image_folder):
        # 获取图像列表并按文件名排序
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()

        # 读取第一个图像以获取尺寸信息
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        size = (width, height)

        # 初始化视频编写器
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        # 将图像帧添加到视频中
        for image in images:
            img_path = os.path.join(image_folder, image)
            img = cv2.imread(img_path)
            video.write(img)

        # 完成视频编写
        video.release()
