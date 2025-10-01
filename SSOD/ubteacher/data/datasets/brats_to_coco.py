import os
import json
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

def get_bbox_from_mask(mask):
    """
    Extract bounding box from binary mask
    Args:
        mask: 2D numpy array of binary mask
    Returns:
        [x, y, width, height] in COCO format
    """
    # Convert to binary mask (threshold > 10)
    binary_mask = (mask > 10).astype(np.uint8)
    
    # Find non-zero points
    y_indices, x_indices = np.where(binary_mask > 0)
    
    if len(y_indices) == 0:
        return None
    
    # Get bounding box coordinates
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)
    
    # Convert to COCO format [x, y, width, height]
    bbox = [
        float(x_min),
        float(y_min),
        float(x_max - x_min),
        float(y_max - y_min)
    ]
    
    return bbox

def create_coco_dataset(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, output_dir, label_ratio=0.05):
    """
    Create COCO format annotations with labeled/unlabeled split
    Args:
        train_img_dir: Directory containing training images
        train_mask_dir: Directory containing training masks
        val_img_dir: Directory containing validation images
        val_mask_dir: Directory containing validation masks
        output_dir: Directory to save JSON files
        label_ratio: Ratio of labeled data in training set
    """
    # Get training files
    train_files = [f for f in os.listdir(train_img_dir) if f.endswith('.nii.gz')]
    
    # Split training files into labeled and unlabeled
    labeled_files, unlabeled_files = train_test_split(
        train_files, 
        test_size=(1-label_ratio),
        random_state=42
    )
    
    # Initialize COCO format dictionaries
    coco_format_template = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "tumor",
                "supercategory": "tumor"
            }
        ]
    }
    
    train_label = coco_format_template.copy()
    train_unlabel = coco_format_template.copy()
    val = coco_format_template.copy()
    
    # Function to process images and create annotations
    def process_files(files, img_dir, mask_dir, coco_dict, include_annotations=True):
        image_id = len(coco_dict["images"])
        annotation_id = len(coco_dict["annotations"])
        
        for file in files:
            # Load mask
            mask_path = os.path.join(mask_dir, file)
            mask_nii = nib.load(mask_path)
            mask_data = mask_nii.get_fdata()
            
            # Process each slice
            for slice_idx in range(mask_data.shape[2]):
                slice_data = mask_data[:, :, slice_idx]
                
                # Skip if no tumor in this slice
                if np.sum(slice_data > 10) == 0:
                    continue
                
                # Create image entry
                image_info = {
                    "id": image_id,
                    "file_name": f"{os.path.splitext(file)[0]}_slice_{slice_idx}.png",
                    "width": int(mask_data.shape[1]),
                    "height": int(mask_data.shape[0])
                }
                coco_dict["images"].append(image_info)
                
                # Add annotation if required
                if include_annotations:
                    bbox = get_bbox_from_mask(slice_data)
                    if bbox is not None:
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0
                        }
                        coco_dict["annotations"].append(annotation)
                        annotation_id += 1
                
                image_id += 1
    
    # Process labeled training data
    process_files(labeled_files, train_img_dir, train_mask_dir, train_label, True)
    
    # Process unlabeled training data
    process_files(unlabeled_files, train_img_dir, train_mask_dir, train_unlabel, False)
    
    # Process validation data
    val_files = [f for f in os.listdir(val_img_dir) if f.endswith('.nii.gz')]
    process_files(val_files, val_img_dir, val_mask_dir, val, True)
    
    # Save annotations
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "instances_train_label.json"), 'w') as f:
        json.dump(train_label, f)
    
    with open(os.path.join(output_dir, "instances_train_unlabel.json"), 'w') as f:
        json.dump(train_unlabel, f)
    
    with open(os.path.join(output_dir, "instances_val.json"), 'w') as f:
        json.dump(val, f)
    
    print(f"Created labeled training annotations with {len(train_label['images'])} images and {len(train_label['annotations'])} annotations")
    print(f"Created unlabeled training annotations with {len(train_unlabel['images'])} images")
    print(f"Created validation annotations with {len(val['images'])} images and {len(val['annotations'])} annotations")

def main():
    # Example usage
    train_img_dir = "path/to/training/images"
    train_mask_dir = "path/to/training/masks"
    val_img_dir = "path/to/validation/images"
    val_mask_dir = "path/to/validation/masks"
    output_dir = "datasets/brats21/annotations"
    
    create_coco_dataset(
        train_img_dir=train_img_dir,
        train_mask_dir=train_mask_dir,
        val_img_dir=val_img_dir,
        val_mask_dir=val_mask_dir,
        output_dir=output_dir,
        label_ratio=0.05  # 5% of training data will be labeled
    )

if __name__ == "__main__":
    main()