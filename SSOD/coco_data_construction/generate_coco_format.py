import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def generate_coco_json(abnormal_img_dir, mask_dir, output_json_path):
    """
    生成COCO格式的数据集JSON文件
    
    Args:
        abnormal_img_dir: 含肿瘤图像目录
        mask_dir: 肿瘤掩码目录
        output_json_path: 输出JSON文件路径
    """
    
    # 初始化COCO格式数据结构
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "tumor",
                "supercategory": "abnormal"
            }
        ]
    }
    
    image_id = 0
    annotation_id = 0
    
    # 处理异常样本
    print("Processing abnormal samples...")
    abnormal_images = sorted([f for f in os.listdir(abnormal_img_dir) if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']])
    
    for img_name in tqdm(abnormal_images):
        # 读取图像获取尺寸
        img_path = os.path.join(abnormal_img_dir, img_name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        
        # 添加图像信息
        coco_format["images"].append({
            "id": image_id,
            "file_name": img_name,
            "height": height,
            "width": width
        })
        
        # 处理对应的mask
        mask_name = img_name.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        
        annotation_added = False  # Track if we added a real annotation
        
        if os.path.exists(mask_path):
            # 读取并二值化mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            
            # 找到非零像素的边界框
            if np.any(binary_mask):
                rows = np.any(binary_mask, axis=1)
                cols = np.any(binary_mask, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                
                # 计算bbox参数
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                
                # 添加标注信息
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x_min), float(y_min), float(bbox_width), float(bbox_height)],
                    "area": float(bbox_width * bbox_height),
                    "iscrowd": 0
                })
                
                annotation_id += 1
                annotation_added = True
        
        # If no valid annotation was added, add a dummy annotation
        # This ensures every image has at least one annotation for SSOD pipeline
        if not annotation_added:
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [0.0, 0.0, 0.0, 0.0],  # Dummy bbox
                "area": 0.0,
                "iscrowd": 0
            })
            annotation_id += 1
        
        image_id += 1
    
    # 保存JSON文件
    print(f"Saving JSON file to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4)
    
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")

# 使用示例
if __name__ == "__main__":
    # For training data
    abnormal_img_dir = "/path/to/images"  # 包含肿瘤的图像目录
    mask_dir = "/path/to/masks"  # 伪标签掩码目录
    output_json_path = "/path/to/instance.json"      # 输出JSON文件路径
    
    generate_coco_json(
        abnormal_img_dir=abnormal_img_dir,
        mask_dir=mask_dir,
        output_json_path=output_json_path
    )
