import cv2
import numpy as np
import supervision as sv
import time
import torch
import torchvision
import os
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor



        
GROUNDING_DINO_CONFIG_PATH = "/home/why/SAM/Grounded-Segment-Anything-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/home/why/SAM/Grounded-Segment-Anything-main/groundingdino_swint_ogc.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CLASSES = ["label", "door", "railing", "shading_device", "window"]
CLASSES = ["label", "door", "railing", "window"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

for i in range(0, 8):    
    # 设置路径
    process_images_folder = f'./1216/img/{i}/'
    # 获取所有图像文件
    process_image_files = sorted(os.listdir(process_images_folder))
    for base_image_file in process_image_files:
    # 更新基准图像路径
        SOURCE_IMAGE_PATH = os.path.join(process_images_folder, base_image_file)
        # load image
        image = cv2.imread(SOURCE_IMAGE_PATH)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=BOX_THRESHOLD
        )
        print('detections: ', detections)
        # detections: xyxy; mask; confidence; class_id; tracker_id


        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # save the annotated grounding dino image
        # cv2.imwrite("groundingdino_annotated_image_17.jpg", annotated_frame)


        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")



        # 计算segment过程的时间
        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        print(labels)
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        os.makedirs(f'./1216/mask_annotated/{i}', exist_ok=True)
        cv2.imwrite(f'./1216/mask_annotated/{i}/{base_image_file}.png', annotated_image)


        mask_result = detections.mask
        print(len(mask_result))
        label_result = detections.class_id
        output_folder = './1216/instance_result'
        if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        dir_name = os.path.split(SOURCE_IMAGE_PATH)[1]
        print(dir_name)
        father_name = os.path.split(os.path.split(SOURCE_IMAGE_PATH)[0])[1]
        print(father_name)
        # 保存mask结果，以及对应的label
        for j in range(len(mask_result)):
            mask = mask_result[j]
            label = label_result[j]
            if label == 0:
                label_str = 'label'
            elif label == 1:
                label_str = 'door'
            elif label == 2:
                label_str = 'railing'
            elif label == 3:
                label_str = 'window'
            mask = mask.astype(np.uint8)
            mask = mask * 255
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask_path = os.path.join(output_folder, father_name, dir_name, f'{label_str}_{j}.jpg')
            print(mask_path)
            if not os.path.exists(os.path.split(mask_path)[0]):
                os.makedirs(os.path.split(mask_path)[0])
            cv2.imwrite(mask_path, mask)
