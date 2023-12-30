import os
import shutil
from PIL import Image, ImageOps
import numpy as np


def read_image(path, image_size=500):
    img = Image.open(path)
    w, h = img.size
    long = max(w, h)
    w, h = int(w / long * image_size), int(h / long * image_size)
    img = img.resize((w, h), Image.ANTIALIAS)
    delta_w, delta_h = image_size - w, image_size - h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(img, padding)


def extract_files(source_folder, total_folder, destination_folder):
    # 获取总文件夹中的所有文件名
    total_files = os.listdir(total_folder)

    # 遍历每个文件夹
    for root, dirs, files in os.walk(source_folder):
        for foldername in dirs:
            source_folder_path = os.path.join(root, foldername)

            # 创建目标文件夹中的相应子文件夹
            destination_folder_path = os.path.join(destination_folder, os.path.relpath(source_folder_path, source_folder))
            os.makedirs(destination_folder_path, exist_ok=True)

            # 遍历文件夹中的文件
            for filename in os.listdir(source_folder_path):
                source_file_path = os.path.join(source_folder_path, filename)

                # 如果文件名在总文件夹中存在
                if filename in total_files:
                    # 将文件复制到目标文件夹中的相应子文件夹
                    ori_filename = os.path.join(total_folder, filename)
                    destination_file_path = os.path.join(destination_folder_path, filename)
                    # shutil.copy(ori_filename, destination_file_path)
                    read_image(ori_filename).save(destination_file_path)
                    print(f"复制文件: {filename} 到 {destination_folder_path}")

# 用法示例
source_folder = "../../../autodl-tmp/masked_4K_fold/"  # 源文件夹路径
total_folder = "../../archive/boneage-training-dataset"    # 总文件夹路径
destination_folder = "../../../autodl-tmp/ori_4K_fold/"  # 目标文件夹路径

extract_files(source_folder, total_folder, destination_folder)
