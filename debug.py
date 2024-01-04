import cv2
import numpy as np
from numba import cuda
import time
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil
import torch






def debug(image_path, mask_folder):
    # 读取原图像
    original_image = cv2.imread(image_path)

    # 初始化叠加后的图像
    overlay_image = original_image.copy()

    mask_files = sorted(os.listdir(mask_folder))
    for mask_file in mask_files:
        if mask_file.endswith(('.JPG', '.jpeg', '.png')):
            mask_path = os.path.join(mask_folder, mask_file)
            
            # 读取掩膜
            mask = cv2.imread(mask_path)

            # 提取白色区域
            white_area = np.all(mask > 0 , axis=-1)
            print(white_area)
            # 设置透明度
            random_color = np.random.randint(0, 255, 3)
            mask[white_area] = random_color
            overlay_image[white_area] = original_image[white_area] * 0.7 + mask[white_area] * 0.3


    # 保存结果
    cv2.imwrite('./mask_result/0.png', overlay_image)

if __name__ == '__main__' :
    path = './picsrc_task2/img/0/03_DSC04088.JPG'
    folder = './1120_component_sup/0/03_DSC04088.JPG'
    # folder = './1120/imagecomponentnew/9/03_DSC04084.JPG'
    debug(path, folder)