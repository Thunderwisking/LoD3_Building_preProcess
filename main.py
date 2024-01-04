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

# 读取图像
# image_base = load_image('./picsrc/img/0/01_DSC04067.JPG').cuda()
# image_toprocess = load_image('./picsrc/img/0/04_DSC05167.JPG').cuda()
# image_base = load_image('../image9/base.JPG').cuda()
# image_toprocess = load_image('../image9/process.JPG').cuda()
# # 读取点txt文件
# with open('../image9/process.txt', 'r') as file:
#     lines = file.readlines()

# 计算映射后的点
# points_coordinates = []
# start = time.time()
# points_coordinates = mp.compute_mapping(image_base, image_toprocess, lines)
# end = time.time()
# print('time cost: ', end - start)


def batch_mapping(base_image_path, process_images_folder, points_folder, points_3d_folder, output_txt, output_txt_3d, output_txt_result):
    # 读取基准图像
    # image_base = load_image(base_image_path).cuda()
    # 获取所有图像文件和对应的点文件
    image_files = sorted(os.listdir(process_images_folder))
    points_files = sorted(os.listdir(points_folder))
    points_files_3d = sorted(os.listdir(points_3d_folder))
    result_list = []
    with open(output_txt, 'a') as output_file:
        # 遍历每个图像
        for image_file, points_file in zip(image_files, points_files):
            # 判断是否有对应的3d点文件
            base_name_2d = points_file[-12:-8]
            print('base name 2d:', base_name_2d)
            matching_3d_file = next((file_3d for file_3d in points_files_3d if base_name_2d in file_3d), None)
            print('matching 3d:', matching_3d_file)
            if matching_3d_file:
                # 构建图像路径和点文件路径
                image_path = os.path.join(process_images_folder, image_file)
                points_path_3d = os.path.join(points_3d_folder, matching_3d_file)
                # print(lines)
            # 计算映射后的点
            match_points_cnt = 0
            # if base_image_path == image_path:
            #     continue
            points_coordinates, match_points_cnt = mp.compute_mapping(base_image_path, image_path, points_file, points_folder)
            # 写入结果到输出文件
            if len(points_coordinates) == 0:
                continue
            result_list.append((base_image_path[:], image_file, match_points_cnt))
            with open(points_path_3d, 'r') as file_3d:
                lines_3d = file_3d.readlines()

            # 移除首行计数
            lines_3d = lines_3d[1:]
            with open(output_txt_3d, 'a') as output_file_3d:
                # output_file_3d.write(f"\n{image_path}\n")
                for line in lines_3d:
                    output_file_3d.write(line)

            # output_file.write(f"\n{image_path}\n")
            for x, y in points_coordinates:
                output_file.write(f"{x} {y}\n")
        
        result_list.sort(key=lambda x: x[2], reverse=True)
        # 写入匹配结果到输出文件
        with open(output_txt_result, 'w') as output_file:
            for result in result_list:
                output_file.write(f"{result[0]} <- {result[1]}  points: {result[2]}\n")

def component_mapping():
    start = time.time()
    for i in range(0, 8):    
        # 设置路径
        process_images_folder = f'./1216/img/{i}/'
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
    # base_image_path = './picsrc_task2/img/9/03_DSC04084.JPG'
    # image_path = './picsrc_task2/img/9/04_DSC04088.JPG'
    # cnt = mp.compute_component_mapping(base_image_path, image_path)
    # print(cnt)

def point_mapping():
    start = time.time()
    for i in range(0,8):    
        # 设置路径
        # process_images_folder = f'./picsrc_task2/img/{i}/'
        # points_folder = f'./picsrc_task2/2d/{i}/'
        # points_3d_folder = f'./picsrc_task2/3d/{i}/'
        # 1216
        process_images_folder = f'./1216/img/{i}/'
        points_folder = f'./1216/2d/{i}/'
        points_3d_folder = f'./1216/3d/{i}/'
        # 获取所有图像文件
        process_image_files = sorted(os.listdir(process_images_folder))
        for base_image_file in process_image_files:
            # 更新基准图像路径
            base_image_path = os.path.join(process_images_folder, base_image_file)
            # 设置输出文件路径
            output_txt = f"./1216/to_2d/{i}/map/to_{base_image_file}.txt"
            output_txt_3d = f"./1216/to_3d/{i}/map/to_{base_image_file}.txt"
            output_txt_result = f"./1216/to_2d/{i}/cnt/result_{base_image_file}.txt"
            # 确保输出文件夹存在
            os.makedirs(os.path.dirname(output_txt), exist_ok=True)

            # 确保3D输出文件夹存在
            os.makedirs(os.path.dirname(output_txt_3d), exist_ok=True)

            # 确保结果输出文件夹存在
            os.makedirs(os.path.dirname(output_txt_result), exist_ok=True)
            # 批量映射和追加
            batch_mapping(base_image_path, process_images_folder, points_folder, points_3d_folder, output_txt, output_txt_3d, output_txt_result)
    end = time.time()
    print('time cost: ', end - start)

def component_fullfill():
    # base_image_path = './picsrc_task2/img/9/03_DSC04084.JPG'
    # image_path = './picsrc_task2/img/9/04_DSC04088.JPG'
    # mp.component_fullfill(base_image_path, image_path)
    
    base_image_path = './1216/img/7/00_DSC03957.JPG'
    image_path = './1216/img/7/'
    process_image_files = sorted(os.listdir(image_path))
    print(process_image_files)
    for image_path1 in process_image_files:
        print(image_path1)
        image_path1 = os.path.join(image_path, image_path1)
        print(image_path1)
        mp.component_fullfill(base_image_path, image_path1)
    return

def mask_semantic():
    base_image_path = './1216/img/7/00_DSC03957.JPG'
    img_path = os.path.split(base_image_path)[1]
    father = os.path.split(os.path.split(base_image_path)[0])[1]
    mask_path = os.path.join('./1216/instance_result',father)
    component_path = os.path.join('./1216_component_sup', father, img_path)
    print(mask_path, component_path)
    mp.label_component(mask_path, component_path)



if __name__ == "__main__":
    # component_mapping()
    # point_mapping()
    # component_fullfill()
    mask_semantic()

    # base_image_path = './picsrc_task2/img/9/03_DSC04084.JPG'
    # process_images_folder = './picsrc_task2/img/9/'
    # points_folder = './picsrc_task2/2d/9/'
    # output_txt = './picsrc_task2/2d/9/to4084.txt'
    # output_txt_3d = './picsrc_task2/3d/9/to4084.txt'
    # output_txt_result = './picsrc_task2/2d/9/result.txt'
    # points_3d_folder = './picsrc_task2/3d/9/'
    # # 批量映射和追加
    # batch_mapping(base_image_path, process_images_folder, points_folder, output_txt)



# # 绘制变换后的图像
# image_with_points = cv2.imread('../image9/base.JPG')
# for point in points_coordinates:
#     x, y = point
#     x = int(x)
#     y = int(y)
#     cv2.circle(image_with_points, (x, y), 5, (0, 255, 0), -1)  # 在图像上绘制绿色点
# image_with_points_rgb1 = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)
# # 显示带有点的图像
# plt.imshow(image_with_points_rgb1)
# plt.axis('off')  # 不显示坐标轴
# plt.show()