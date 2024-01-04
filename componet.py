import cv2
import numpy as np
from numba import cuda
import time
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mapping as mp
import os



if __name__ == "__main__":
    start = time.time()
    for i in range(0, 11):    
        # 设置路径
        process_images_folder = f'./picsrc_task2/img/{i}/'
        # 获取所有图像文件
        process_image_files = sorted(os.listdir(process_images_folder))
        for base_image_file in process_image_files:
            # 更新基准图像路径
            for image_file in process_image_files:
                base_image_path = os.path.join(process_images_folder, base_image_file)
                image_path = os.path.join(process_images_folder, image_file)
                cnt = mp.compute_component_mapping(base_image_path, image_path)
                if cnt != None:
                    with open('./debug.txt', 'w') as output_file:
                        output_file.write(f'base image: {base_image_path}, to image: {image_path}, match points: {cnt}, mask written successfully!\n')
                else:
                    with open('./debug.txt', 'w') as output_file:
                        output_file.write(f'base image: {base_image_path}, to image: {image_path}, pair failed!!!\n')
    end = time.time()
    print('time cost: ', end - start)

