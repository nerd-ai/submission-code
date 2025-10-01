from PIL import Image
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats
import numpy as np
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2
import random

# Build model config
opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

# Load model from pretrained weights
pretrained_pth = '/path/to/your/model_checkpoint.pth'  # set your model checkpoint path here

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)


def get_random_prompt(seed=None):
    """Get a random prompt from multiple choices using optional seed"""
    # prompt_choices = [
    #     ['Lower-grade glioma in brain MRI'],
    #     ['malignant tumor in breast ultrasound'],
    #     ['Melanoma in skin Dermoscopy'],
    #     ['Cystoid macular edema in retinal OCT'],
    #     ['Neoplastic polyp in colon Endoscope'],
    #     ['liver in abdomen CT'],
    #     ['kidney in kidney ultrasound']
    # ]
    # prompt_choices = [
    #          ['Lower-grade glioma in brain MRI'],
    #          ['Low-grade glioma as observed in T2-weighted brain MRI'],
    #          ['Low-grade glioma delineated in cranial MRI.'],
    #          ['Low-grade glioma presented in a brain MRI scan'],
    #          ['Low-grade glioma imaged in contrast-enhanced brain MRI.'],
    #          ['Low-grade glioma identified through cerebral MRI.'],
    #          ['Low-grade glioma visual pattern in high-resolution brain MRI.'],
    #          ['Low-grade glioma detected via structural brain MRI.'],
    #          ['Low-grade glioma highlighted in a diagnostic brain MRI.'],
    #          ['Low-grade glioma manifestation in functional MRI of the brain.']
    # ]
    
    # prompt_choices =[ 
    #         ['Low-grade glioma visualized in brain MRI.'],
    #         ['Low-grade glioma depicted on magnetic resonance imaging of the brain.'],
    #         ['Low-grade glioma presented in a brain MRI scan.'],
    #         ['Low-grade glioma identified through cerebral MRI.'],
    #         ['Low-grade glioma evident in axial brain MR imaging.'],
    #         ['Low-grade glioma as observed in T2-weighted brain MRI.'],
    #         ['Low-grade glioma delineated in cranial MRI.'],
    #         [ 'Low-grade glioma localized in neuroimaging using MRI.'],
    #         ['Low-grade glioma appearing in magnetic resonance imaging of the brain.'],
    #         ['Low-grade glioma detected via structural brain MRI.'],
    #         ['Low-grade glioma observed in MR brain scans.'],
    #         ['Low-grade glioma captured in multiplanar brain MRI.'],
    #         ['Low-grade glioma imaged in contrast-enhanced brain MRI.'],
    #         ['Low-grade glioma segmented on brain magnetic resonance imaging.'],
    #         ['Low-grade glioma highlighted in a diagnostic brain MRI.'],
    #         ['Low-grade glioma visual pattern in high-resolution brain MRI.'],
    #         ['Low-grade glioma as seen in a sagittal slice of brain MRI.'],
    #         ['Low-grade glioma visible on coronal brain MRI images.'],
    #         ['Low-grade glioma represented in preoperative brain MRI.'],
    #         ['Low-grade glioma manifestation in functional MRI of the brain.'] 
    # ]
    
    prompt_choices = [
            ['Liver depicted on abdominal CT scan.'],
            ['Liver as visualized in computed tomography of the abdomen'],
            ['Liver region segmented from an abdominal CT image'],
            ['Liver captured on contrast-enhanced abdominal CT'],
            ['Liver located in axial abdominal CT imaging'],
            ['Liver anatomy delineated in abdominal CT scan'],
            ['Liver illustrated in a cross-sectional CT of the abdomen'],
            ['Liver shown in a diagnostic abdominal CT slice'],
            ['Liver appearance on a portal venous phase abdominal CT'],
            ['Liver structures in abdomen-focused CT imaging'],
            ['Liver mapped in multi-phase CT of the upper abdomen'],
            ['Liver highlighted in abdominopelvic CT acquisition'],
            ['Liver identified in a transverse CT scan of the abdomen'],
            ['Liver segmented in a CT dataset of the abdominal cavity'],
            ['Liver represented in a DICOM abdominal CT series'],
            ['Liver visualized using computed tomography of the liver and abdomen'],
            ['Liver displayed in a radiological CT scan of the abdomen'],
            ['Liver assessed through abdominal CT imaging'],
            ['Liver extracted from abdominal tomographic imaging'],
            ['Liver portrayed in a volumetric CT scan of the abdomen']
    ]
    
    # prompt_choices = [ 
    #         ['Kidney as visualized in renal ultrasound imaging'],
    #         ['Kidney depicted on grayscale ultrasound of the kidney'],
    #         ['Kidney outlined in a longitudinal renal ultrasound scan'],
    #         ['Kidney observed in a sonographic image of the renal system'],
    #         ['Kidney shown in B-mode ultrasound of the kidney region'],
    #         ['Kidney identified in a diagnostic ultrasound scan of the kidney'],
    #         ['Kidney imaged through abdominal sonography focused on the kidneys'],
    #         ['Kidney represented in real-time renal ultrasonographic imaging'],
    #         ['Kidney delineated in high-resolution renal ultrasound data'],
    #         ['Kidney visualized in a transverse ultrasound slice of the kidney'],
    #         ['Kidney assessed via non-invasive renal ultrasound examination'],
    #         ['Kidney captured in an axial ultrasound scan of the renal area'],
    #         ['Kidney appearing in ultrasound acquisition of the kidneys'],
    #         ['Kidney segmented from a focused ultrasound of the renal cortex'],
    #         ['Kidney displayed in sonographic imaging of the urinary tract'],
    #         ['Kidney highlighted in grayscale sonography of the renal parenchyma'],
    #         ['Kidney imaged using handheld ultrasonography in the kidney region'],
    #         ['Kidney visualized on real-time renal sonographic imaging'],
    #         ['Kidney featured in point-of-care ultrasound (POCUS) of the kidneys'],
    #         ['Kidney rendered in 2D ultrasound imaging of the renal zone']          
    # ]    
    
    if seed is not None:
        random.seed(seed)
    
    return random.choice(prompt_choices)


def process_folder(model, input_dir, output_dir, random_seed=None):
    """Process all images and save only probability masks with random prompt selection"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in tqdm(image_files):
        # Get random prompt for this image
        prompt = get_random_prompt(random_seed)
        
        # Read image
        image_path = os.path.join(input_dir, img_name)
        image = Image.open(image_path).convert('RGB')
        
        # Get prediction
        pred_mask = interactive_infer_image(model, image, prompt)
        
        # Process prediction mask
        mask = pred_mask[0]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        mask = mask.squeeze()
        
        # Ensure mask dimensions are correct
        if mask.shape != image.size[::-1]:
            mask = cv2.resize(mask, image.size[::-1])
            
        # Save using np.save directly for floating point values
        npy_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy')
        np.save(npy_path, mask)
        
        # Optionally save prompt info for debugging
        prompt_info_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '_prompt.txt')
        with open(prompt_info_path, 'w') as f:
            f.write(f"Prompt used: {prompt[0]}\n")
            if random_seed is not None:
                f.write(f"Random seed: {random_seed}\n")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    input_dir = "/path/to/the_original_image/"
    output_dir = "/path/to/the_confidence_map_npy_temperature_t1/"

    # Set random seed for reproducibility (optional)
    random_seed = 42
    
    process_folder(model, input_dir, output_dir, random_seed)