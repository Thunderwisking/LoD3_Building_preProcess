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





def judge_semantic(component_path, to_pair_dir, flag_o):
    # 读取原图像
    flag = flag_o
    component_origin = cv2.imread(component_path)
    component = cv2.cvtColor(component_origin, cv2.COLOR_BGR2GRAY)
    cnt = 0
    for mask_file in os.listdir(to_pair_dir):
        mask_path = os.path.join(to_pair_dir, mask_file)
        # 读取掩膜
        mask_origin = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask_origin, cv2.COLOR_BGR2GRAY)
        occupy = compute_occupy(component, mask)
        if occupy > 0.4:
            label = mask_file.split('_')[0]
            if label == 'label':
                flag[0] = flag[0] + 1
                cnt = cnt + 1
            elif label == 'door':
                flag[1] = flag[1] + 1
                cnt = cnt + 1
            elif label == 'railing':
                flag[2] = flag[2] + 1
                cnt = cnt + 1   
            elif label == 'window':
                flag[3] = flag[3] + 1
                cnt = cnt + 1
        # 计算component和mask的occupy值
    return flag


def label_write(flag, component_to_label_path):
    target_dir = './1216/semetic_result/'
    print(component_to_label_path)
    origin_name, name1, name2= component_to_label_path.split('/')[4], component_to_label_path.split('/')[2], component_to_label_path.split('/')[3]
    if origin_name.startswith('t_'):
        origin_name1 = origin_name[2:13] + '.JPG'
        lenth = len(os.listdir(os.path.join('./1216/1216_result', name1, origin_name1)))
    target_dir = os.path.join(target_dir, name1, name2, name2)
    print(component_to_label_path)
    lenth = len(os.listdir(os.path.join('./1216/1216_result', name1, name2)))
    print(flag)
    print('len: ', lenth)
    # 如果不存在则创建文件夹
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if flag == [0, 0, 0, 0]:
        target_dir = os.path.join(target_dir, f'undifined_{origin_name}')
        shutil.copy(component_to_label_path, target_dir) 
    else:
        max_index = np.argmax(flag)
        if(max_index == 0):
            if(flag[max_index]>(lenth/2)):
                target_dir = os.path.join(target_dir, f'covering_{origin_name}')
                shutil.copy(component_to_label_path, target_dir) 
            else:
                target_dir = os.path.join(target_dir, f'undifined_{origin_name}')
                shutil.copy(component_to_label_path, target_dir) 
        if(max_index == 1):
            if(flag[max_index]>(lenth/2)):
                target_dir = os.path.join(target_dir, f'door_{origin_name}')
                shutil.copy(component_to_label_path, target_dir)
            else:
                target_dir = os.path.join(target_dir, f'undifined_{origin_name}')
                shutil.copy(component_to_label_path, target_dir) 
        if(max_index == 2):
            if(flag[max_index]>(lenth/2)):
                target_dir = os.path.join(target_dir, f'railing_{origin_name}')
                shutil.copy(component_to_label_path, target_dir) 
            else:
                target_dir = os.path.join(target_dir, f'undifined_{origin_name}')
                shutil.copy(component_to_label_path, target_dir) 
        if(max_index == 3):
            if(flag[max_index]>(lenth/2)):
                target_dir = os.path.join(target_dir, f'window_{origin_name}')
                shutil.copy(component_to_label_path, target_dir) 
            else:
                target_dir = os.path.join(target_dir, f'undifined_{origin_name}')
                shutil.copy(component_to_label_path, target_dir) 
    return 


def label_component(mask_path, component_path):
    for component in os.listdir(component_path):
        if component.endswith(('.JPG', '.jpeg', '.png')):
            component_to_label_path = os.path.join(component_path, component)
            print(component_to_label_path)
            if not component.startswith('t_'):              
                flag = [0, 0, 0, 0]
                to_pair_dir = os.path.join(mask_path, os.path.basename(component_path))
                flag = judge_semantic(component_to_label_path, to_pair_dir, flag)
                other0, other1 = component_to_label_path.split('/')[2], component_to_label_path.split('/')[3]
                search_dir = os.path.join('./1216/1216_result/', other0, other1)
                print(len(os.listdir(search_dir)))
                if len(os.listdir(search_dir)) == 1:
                    label_write(flag, component_to_label_path)
                else:
                    for dir in os.listdir(search_dir):
                        if dir == os.path.basename(component_path):
                            continue
                        else:
                            component_to_label_new_path = os.path.join('./1216/1216_result/', other0, other1, dir, component)
                            # print('new:', component_to_label_new_path)
                            to_pair_dir_new = os.path.join(mask_path, os.path.basename(dir))
                            # print('to pair new ', to_pair_dir_new)
                            flag = judge_semantic(component_to_label_new_path, to_pair_dir_new, flag)
                    label_write(flag, component_to_label_path)
            # 处理别的照片补充到最佳照片上的组件
            else:
                flag = [0, 0, 0, 0]
                component_s = component[14:]
                print(component_s)
                basename = component[2:13]
                print(basename)
                component_to_label_s_path = os.path.join('./1216/imagecomponentnew', component_to_label_path.split('/')[2], f'{basename}.JPG', component_s)
                print(component_to_label_s_path)
                flag = [0, 0, 0, 0]
                to_pair_dir1 = os.path.join(mask_path, f'{basename}.JPG')
                print('to_pair_dir1 ', to_pair_dir1)
                flag = judge_semantic(component_to_label_s_path, to_pair_dir1, flag)
                other = component_to_label_s_path.split('/')[3]
                search_dir = os.path.join('./1216/1216_result/', other, f'{basename}.JPG')
                print(search_dir)
                print((os.listdir(search_dir)))
                if len(os.listdir(search_dir)) == 1:
                    label_write(flag, component_path)
                else:
                    for dir in os.listdir(search_dir):
                        if dir == os.path.basename(component_path):
                            continue
                        else:
                            component_to_label_new_path = os.path.join('./1216/1216_result/', other, f'{basename}.JPG', dir, component_s)
                            print('new:', component_to_label_new_path)
                            to_pair_dir_new = os.path.join(mask_path, os.path.basename(dir))
                            print('to pair new ', to_pair_dir_new)
                            flag = judge_semantic(component_to_label_new_path, to_pair_dir_new, flag)
                    label_write(flag, component_to_label_path)
                
        else:
            continue
            

def load_mkpt(base_image_path, image_path):
    image_base = load_image(base_image_path).cuda()
    image_toprocess = load_image(image_path).cuda()
    # 创建映射后的图像
    base_image_height, base_image_width = image_base.size(1), image_base.size(2)
    image_height, image_width =  image_toprocess.size(1), image_toprocess.size(2)
    # extract local features
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
    feats0 = extractor.extract(image_base)  # auto-resize the image, disable with resize=None/
    feats1 = extractor.extract(image_toprocess)
    # print(feats0)
    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints']  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints']  # coordinates in image #1, shape (K,2)
    # 获取两组对应的匹配点 并存储为numpy形式
    m_kpts0 = points0[matches[..., 0]]
    m_kpts1 = points1[matches[..., 1]]
    m_kpts0 = m_kpts0.cpu().numpy()
    m_kpts1 = m_kpts1.cpu().numpy()
    return m_kpts0, m_kpts1, base_image_height, base_image_width, image_height, image_width



def find_componet_center(image_to_find_path):
    image_to_find = cv2.imread(image_to_find_path, cv2.IMREAD_GRAYSCALE)
    # 寻找白色区域的轮廓
    _, contours, _ = cv2.findContours(image_to_find, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的矩
        M = cv2.moments(contour)
        # 防止分母为零
        if M["m00"] != 0:
            # 计算中心点坐标
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            return center_x, center_y

def create_result_png(file_path, to_process_height, to_process_width, H, output_path, flag):
    print(to_process_height, to_process_width)
    _, target_name = os.path.split(file_path)
    target_path = os.path.join(output_path, target_name)
    print(target_name)
    component_base = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    height, width = component_base.shape
    warped_mask = np.zeros((to_process_height, to_process_width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # 如果当前像素是白色
            if component_base[y, x] == 255:
                # 应用单应性变换
                src_point = np.array([[x, y, 1]])
                dst_point = np.dot(H, src_point.T).T
                dst_x, dst_y, dst_w = dst_point[0]

                # 归一化坐标
                dst_x = int(dst_x / dst_w)
                dst_y = int(dst_y / dst_w)

                # 确保新坐标在图像范围内
                if 0 <= dst_x < to_process_width and 0 <= dst_y < to_process_height:
                    warped_mask[dst_y, dst_x] = 255
    kernel_size = 10
    # 使用方形的结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filled_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel)
    # _, contours, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 创建一个与 warped_mask 相同大小的全黑图像，用于填充
    # filled_mask = np.zeros_like(warped_mask)
    # # 填充每个轮廓
    # cv2.fillPoly(filled_mask, contours, 255)
    # print(output_path)
    if flag == 1:
        cv2.imwrite(target_path, filled_mask)
    return filled_mask


    # cv2.imshow('Warped Mask', filled_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def component_fullfill(base_image_path, image_path):
    m_kpts0, m_kpts1, base_height, base_width, _, _ = load_mkpt(base_image_path, image_path)
    if(m_kpts0.shape[0] < 400):
        return None
    else:
        # 这里获取的是非最佳底板上的组件儿
        head, name = os.path.split(image_path)
        _, father = os.path.split(head)
        component_pic_dir = os.path.join('./1216/imagecomponentnew', father, name)
        to_process_component_base = os.path.join('./1216/bianjie_maskBW', father, name)
        if not os.path.exists(component_pic_dir):
            print(f"Directory {component_pic_dir} does not exist. Skipping...")
            return None
        # 这里读取的是最佳底板
        head_base, name_base = os.path.split(base_image_path)
        _, father_base = os.path.split(head_base)
        _, name_base = os.path.split(base_image_path)
        base_component_pic_dir = os.path.join('./1216/imagecomponentnew', father_base, name_base)
        output_dir = os.path.join('./1216_component_sup', father_base, name_base)

        # 首先将最佳底板的组件复制到输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            for file in os.listdir(base_component_pic_dir):
                shutil.copy(os.path.join(base_component_pic_dir, file), os.path.join(output_dir, file))
        
        # 遍历非最佳底板的组件
        for file in os.listdir(component_pic_dir):
            # 判断m_kpt1中在file的mask白色区域的个数
            file_path = os.path.join(component_pic_dir, file)
            area, num_to_pair = compute_area(file_path, m_kpts1)
            if(num_to_pair < 5):
                continue
            if(num_to_pair > 5):
                # 计算组件由非最佳到最佳底板的单应性变换
                H = compute_H_new(to_process_component_base, m_kpts1, m_kpts0, image_path, base_image_path)
                mask_image = create_result_png(file_path, base_height, base_width, H, output_dir, 0)
                # 遍历当前输出文件夹中的组件
                flag = 0
                to_judge = []
                for component_temp in os.listdir(output_dir):
                    if component_temp.lower().endswith(('.JPG', '.jpeg', '.png')):
                        # 计算当前组件与当前输出文件夹中的组件的重叠度
                        component_temp_path = os.path.join(output_dir, component_temp)
                        overlap_cur, overlap_best = compute_overlap(mask_image, component_temp_path)
                        # print(overlap_cur)
                        if overlap_cur < 0.4:
                            flag = -100
                            continue
                        else:
                            if overlap_cur == 1:
                                continue
                            else:
                                print(overlap_cur)
                                flag += 1
                                to_judge.append(component_temp_path)          
                # 映射组件与任何组件都不相交 则单独写入
                print('flag: ', flag)
                print(to_judge)
                # if flag == -100:
                #     continue
                if flag < 0:
                    continue
                if flag == 0:
                    original_filename = os.path.basename(file)
                    parent_folder_name = os.path.basename(component_pic_dir)
                    new_folder_name = parent_folder_name[:-4]
                    new_filename = f"t_{new_folder_name}_{original_filename}"
                    output_path = os.path.join(output_dir, new_filename)
                    with open(f'./{output_dir}/log.txt', 'a') as f:
                        f.write(f'simply added: {new_filename}  flag = {flag}\n')
                    cv2.imwrite(output_path, mask_image)
                # 存在相交 逐个判断
                else:
                    for component_temp_path in to_judge:
                        overlap_cur, overlap_best = compute_overlap(mask_image, component_temp_path)
                        if overlap_best > 2/3:
                            # 直接添加当前组件的映射
                            original_filename = os.path.basename(file)
                            parent_folder_name = os.path.basename(component_pic_dir)
                            new_folder_name = parent_folder_name[:-4]
                            new_filename = f"t_{new_folder_name}_{original_filename}"
                            output_path = os.path.join(output_dir, new_filename)
                            with open(f'./{output_dir}/log.txt', 'a') as f:
                                f.write(f'judged to {component_temp_path} then simply added: {new_filename}  flag = {flag} cur = {overlap_cur} best = {overlap_best}\n')
                            cv2.imwrite(output_path, mask_image)
                        else:
                            # 删除最佳组件 -> 添加当前组件的映射
                            os.remove(component_temp_path)
                            original_filename = os.path.basename(file)
                            parent_folder_name = os.path.basename(component_pic_dir)
                            new_folder_name = parent_folder_name[:-4]
                            new_filename = f"t_{new_folder_name}_{original_filename}"
                            output_path = os.path.join(output_dir, new_filename)
                            with open(f'./{output_dir}/log.txt', 'a') as f:
                                f.write(f'deleted: {component_temp_path} added: {new_filename}   flag = {flag} cur = {overlap_cur} best = {overlap_best}\n')
                            success = cv2.imwrite(output_path, mask_image)

# 非最佳组件 -> 最佳组件
def compute_overlap(mask_image, component_temp_path):
    # 读取当前输出文件夹中的组件
    component_temp = cv2.imread(component_temp_path)
    # 将组件转换为灰度图
    component_temp = cv2.cvtColor(component_temp, cv2.COLOR_BGR2GRAY)
    overlap_cur = np.sum(mask_image) - 255 * np.sum(np.logical_and(mask_image, component_temp))
    overlap_cur = overlap_cur / np.sum(mask_image)
    overlap_best = np.sum(component_temp) - 255 * np.sum(np.logical_and(mask_image, component_temp))
    overlap_best = overlap_best / np.sum(component_temp)
    return overlap_cur, overlap_best

def compute_occupy(component, mask):
    overlap_pixels = np.sum((component == 255) & (mask == 255))
    white_pixels_mask = np.sum(mask == 255)
    ratio = overlap_pixels / white_pixels_mask
    return ratio

def compute_area(mask_path, m_kpts1):
    # 计算组件的面积
    total_area = 0
    num_to_pair = 0
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0
    for contour in contours:
        total_area += cv2.contourArea(contour)
    unit = int(np.sqrt(total_area)*0.2)
    kernel = np.ones((unit, unit), np.uint8)
    bigger_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    _, contours1, _ = cv2.findContours(bigger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours1:
        for point in m_kpts1:
            if cv2.pointPolygonTest(contour, tuple(point), False) > 0:
                num_to_pair += 1

    print('total area: ', total_area)
    print('points in mask: ', num_to_pair)
    # cv2.imshow('mask', mask)
    # cv2.imshow('bigger_mask', bigger_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return total_area, num_to_pair


def compute_component_mapping(base_image_path, image_path):
    m_kpts0, m_kpts1, _, _, to_process_height, to_process_width = load_mkpt(base_image_path, image_path)
    if(m_kpts0.shape[0] < 400):
        return None
    else:
        # WARNING： 读取的组件还是base 但是最后的计算结果是base到toprocess！！！！
        head_base, name_base = os.path.split(base_image_path)
        _, father_base = os.path.split(head_base)
        component_pic_dir = os.path.join('./1216/imagecomponentnew', father_base, name_base)
        base_component_dir = os.path.join('./1216/bianjie_maskBW', father_base, name_base)
        if not os.path.exists(component_pic_dir):
            print(f"Directory {component_pic_dir} does not exist. Skipping...")
            return None
        head, name_toprocess = os.path.split(image_path)
        _, father = os.path.split(head)
        _, name_base = os.path.split(base_image_path)
        output_dir = os.path.join('./1216/1216_result', father, name_base, name_toprocess)
        os.makedirs(output_dir, exist_ok=True)    
        for component in os.listdir(component_pic_dir):
            # if component == '41.png':
                if component.lower().endswith(('.JPG', '.jpeg', '.png')):
                    file_path = os.path.join(component_pic_dir, component)
                    print(file_path)
                    # center_x, center_y = find_componet_center(file_path)
                    # 选取邻域特征点计算单应变换
                    # H = compute_H(center_x, center_y, m_kpts0, m_kpts1, base_image_path, image_path)
                    # H = compute_H_new(file_path, m_kpts0, m_kpts1, base_image_path, image_path)
                    H = compute_H_new(base_component_dir, m_kpts0, m_kpts1, base_image_path, image_path)
                    # H_inv = cv2.invert(H)
                    # print(H)
                    mask_image = create_result_png(file_path, to_process_height, to_process_width, H, output_dir, 1)
                    # print(mask_image.shape)
                # mask_debug(base_image_path, image_path, file_path, mask_image)
                

        return m_kpts0.shape[0]


def mask_debug(base_image_path, image_path, file_path, mask_image):
    alpha = 0.3
    _, alpha_channel = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)
    alpha_channel = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
    original_image = cv2.imread(image_path)
    alpha_channel_resized = cv2.resize(alpha_channel, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1 - alpha, alpha_channel_resized, alpha, 0)
    cv2.imwrite('result.png', result)

    mask_origin = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    alpha = 0.3
    _, alpha_channel1 = cv2.threshold(mask_origin, 1, 255, cv2.THRESH_BINARY)
    alpha_channel1 = cv2.merge([alpha_channel1, alpha_channel1, alpha_channel1])
    original_image1 = cv2.imread(base_image_path)
    alpha_channel_resized1 = cv2.resize(alpha_channel1, (original_image1.shape[1], original_image1.shape[0]))
    result1 = cv2.addWeighted(original_image1, 1 - alpha, alpha_channel_resized1, alpha, 0)
    cv2.imwrite('origin.png', result1)

# 0 -> 1
def compute_H_new(file_path, m_kpts0, m_kpts1, base_image_path, image_path):
    functional_points0 = []
    functional_points1 = []
    base_image = cv2.imread(base_image_path)
    target_image = cv2.imread(image_path)
    component = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    print(file_path)
    # cv2.imshow('Original Mask', component)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  
    _, contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0
    for contour in contours:
        total_area += cv2.contourArea(contour)
    unit = int(np.sqrt(total_area)*0.1)
    kernel = np.ones((unit, unit), np.uint8)
    shrinked_mask = cv2.morphologyEx(component, cv2.MORPH_ERODE, kernel)
    # cv2.imshow('Original Mask', component)
    # cv2.imshow('Shrinked Mask', shrinked_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 判断kpts0中的点是否在component中
    for (w, h), (x, y) in zip(m_kpts0, m_kpts1):
        if(shrinked_mask[int(h), int(w)] == 255):
            functional_points0.append([w, h])
            functional_points1.append([x, y])
    #         cv2.circle(base_image, (int(w), int(h)), 5, (0, 0, 255), -1)
    #         cv2.circle(target_image, (int(x), int(y)), 5, (0, 0, 255), -1)
    # cv2.imwrite('base_image.png', base_image)
    # cv2.imwrite('target_image.png', target_image)
    functional_points0 = np.array(functional_points0)
    functional_points1 = np.array(functional_points1)
    # print(':',functional_points0)
    H, _ = cv2.findHomography(functional_points0, functional_points1, cv2.RANSAC, 5.0)
    return H

def compute_H(center_x, center_y, m_kpts0, m_kpts1, base_image_path, image_path):
    functional_points0 = []
    functional_points1 = []
    base_image = cv2.imread(base_image_path)
    target_image = cv2.imread(image_path)
    for radius in range(10, 2000, 10):
        flag = 0
        for (w, h), (x, y) in zip(m_kpts0, m_kpts1):
            distance = ((w - center_x) ** 2 + (h - center_y) ** 2) ** 0.5
            if distance < radius:
                flag = flag + 1
                # print([w,h])
                cv2.circle(base_image, (int(w), int(h)), 5, (0, 0, 255), -1)
                cv2.circle(target_image, (int(x), int(y)), 5, (0, 0, 255), -1)
                functional_points0.append([w, h])
                functional_points1.append([x, y])
                if flag >= 10:
                    print(radius)
                    print(flag)
                    break
        if flag >= 10:
            print(radius)
            print(flag)
            break
        else:
            functional_points0 = []
            functional_points1 = []
    cv2.imwrite('base_image.png', base_image)
    cv2.imwrite('target_image.png', target_image)
    functional_points0 = np.array(functional_points0)
    functional_points1 = np.array(functional_points1)
    # print(':',functional_points0)
    H, _ = cv2.findHomography(functional_points0, functional_points1, cv2.RANSAC, 5.0)
    return H




def compute_mapping(base_image_path, image_path, points_file, points_folder):
    image_base = load_image(base_image_path).cuda()
    image_toprocess = load_image(image_path).cuda()
    # 创建映射后的图像
    base_height, base_width = image_base.shape[:2]
    target_image = np.zeros((base_height, base_width, 3), dtype=np.uint8)
    # extract local features
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
    feats0 = extractor.extract(image_base)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image_toprocess)
    # print(feats0)
    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints']  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints']  # coordinates in image #1, shape (K,2)
    # 获取两组对应的匹配点 并存储为numpy形式
    m_kpts0 = points0[matches[..., 0]]
    m_kpts1 = points1[matches[..., 1]]
    m_kpts0 = m_kpts0.cpu().numpy()
    m_kpts1 = m_kpts1.cpu().numpy()
    print(m_kpts0.shape)
    # 若取到的特征点过少 则返回空值
    print(m_kpts0.shape[0])
    if(m_kpts0.shape[0] < 400):
        lines = []
        return [], m_kpts0.shape[0]
    else:
        points_path = os.path.join(points_folder, points_file)
        print(points_path)
        # 读取点txt文件
        with open(points_path, 'r') as file:
            lines = file.readlines()
        # 直接计算整张图像的Homography矩阵
        H, _ = cv2.findHomography(m_kpts1, m_kpts0, cv2.RANSAC, 5.0)
        transformed_coordinates = []
        origin_coordinates = []
        part_H_coordinates = []
        readline = []
        # 逐行处理待处理 txt 文件中的数据
        for line in lines:
            parts = line.split()
            if len(parts) == 6:
                # 解析坐标
                x, y = float(parts[0]), float(parts[1])
                # 创建一个列向量
                original_point = np.array([[x], [y], [1]])
                # 使用变换矩阵 H 来执行透视变换
                transformed_point = np.dot(H, original_point)
                # 获取变换后的坐标
                transformed_x, transformed_y, w = transformed_point.flatten()
                # 忽略无效数据
                if w != 0:
                    transformed_x /= w
                    transformed_y /= w
                    transformed_coordinates.append((transformed_x, transformed_y))
                    origin_coordinates.append((x, y))
                    readline.append((float(parts[0]), float(parts[1])))
                    # print(readline)
                # 用邻域的方法计算映射
                functional_points0 = []
                functional_points1 = []
                # 选取最近且最少的特征点
                for radius in range(100, 2000, 100):
                    flag = 0
                    for (w, h), (x1, y1) in zip(m_kpts1, m_kpts0):
                        distance = ((w - x) ** 2 + (h - y) ** 2) ** 0.5
                        if distance < radius:
                            flag = flag + 1
                            functional_points1.append([w, h])
                            functional_points0.append([x1, y1])
                    if flag >= 10:
                        break
                    else:
                        # print('flag:', flag)
                        functional_points0 = []
                        functional_points1 = []
                functional_points0 = np.array(functional_points0)
                functional_points1 = np.array(functional_points1)
                # print(':',functional_points0)
                H2, _ = cv2.findHomography(functional_points1, functional_points0, cv2.RANSAC, 5.0)
                original_point1 = np.array([[x], [y], [1]])
                transformed_point1 = np.dot(H2, original_point1)
                transformed_x1, transformed_y1, w1 = transformed_point1.flatten()
                if w1 != 0:
                    transformed_x1 /= w1
                    transformed_y1 /= w1
                    part_H_coordinates.append((transformed_x1, transformed_y1))
        part_H_coordinates1 = np.array(part_H_coordinates)
        transformed_coordinates = np.array(transformed_coordinates)
        print(part_H_coordinates1.shape)
        # print(transformed_coordinates.shape)
        # 保存变换后的坐标到新的 txt 文件
        with open('transformed_coordinates_singleH.txt', 'w') as output_file:
            for x, y in transformed_coordinates:
                output_file.write(f"{x} {y}\n")



        # # visualize
        # image_todraw = cv2.imread('./picsrc/img/0/01_DSC04067.JPG')
        # image_todraw1 = cv2.imread('./picsrc/img/0/04_DSC05167.JPG')
        # height = max(image_todraw.shape[0], image_todraw1.shape[0])
        # combined_image = np.zeros((height, image_todraw.shape[1] + image_todraw1.shape[1], 3), dtype=np.uint8)
        # combined_image[:image_todraw.shape[0], :image_todraw.shape[1]] = image_todraw
        # combined_image[:image_todraw1.shape[0], image_todraw.shape[1]:] = image_todraw1

        # # 绘制特征点和线
        # for pt0, pt1 in zip(m_kpts0, m_kpts1):
        #     pt1[0] += image_todraw.shape[1]  # 调整第二幅图像上的特征点的 x 坐标，使其与第一幅图像对齐
        #     cv2.circle(combined_image, (int(pt0[0]), int(pt0[1])), 5, (0, 0, 255), -1)
        #     cv2.circle(combined_image, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 0), -1)
        #     cv2.line(combined_image, (int(pt0[0]), int(pt0[1])), (int(pt1[0]), int(pt1[1])), (0, 200, 0), 1)

        # # 显示图像
        # cv2.imshow('Matches', combined_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return part_H_coordinates1, m_kpts0.shape[0]
        
    