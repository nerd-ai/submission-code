---
license: cc-by-nc-sa-4.0
datasets:
- microsoft/BiomedParseData
---

This is the official model checkpoint repo for "A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities".

[[`Code`](https://github.com/microsoft/BiomedParse)] [[`Paper`](https://aka.ms/biomedparse-paper)] [[`Demo`](https://microsoft.github.io/BiomedParse/)] [[`Data`](https://huggingface.co/datasets/microsoft/BiomedParseData)]

Biomedical image analysis is fundamental for biomedical discovery in cell biology, pathology, radiology, and many other biomedical domains. BiomedParse is a biomedical foundation model for imaging parsing that can jointly conduct segmentation, detection, and recognition across 9 imaging modalities. Through joint learning, we can improve accuracy for individual tasks and enable novel applications such as segmenting all relevant objects in an image through a text prompt, rather than requiring users to laboriously specify the bounding box for each object.

BiomedParse is broadly applicable, performing image segmentation across 9 imaging modalities.

### Installation
```sh
git clone https://github.com/microsoft/BiomedParse.git
cd BiomedParse

conda create -n biomedparse python=3.9.19
conda activate biomedparse
```

Install Pytorch
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
In case there is issue with detectron2 installation, make sure your pytorch version is compatible with CUDA version on your machine at https://pytorch.org/.

Install dependencies
```sh
pip install -r assets/requirements/requirements.txt
```

### Model Setup
```sh
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

# Build model config
opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

# Load model from pretrained weights
pretrained_pth = 'hf_hub:microsoft/BiomedParse'

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
```

### Segmentation On Example Images
```sh
# RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
image = Image.open('examples/Part_1_516_pathology_breast.png', formats=['png'])
image = image.convert('RGB')
# text prompts querying objects in the image. Multiple ones can be provided.
prompts = ['neoplastic cells', 'inflammatory cells']

# load ground truth mask
gt_masks = []
for prompt in prompts:
    gt_mask = Image.open(f"examples/Part_1_516_pathology_breast_{prompt.replace(' ', '+')}.png", formats=['png'])
    gt_mask = 1*(np.array(gt_mask.convert('RGB'))[:,:,0] > 0)
    gt_masks.append(gt_mask)

pred_mask = interactive_infer_image(model, image, prompts)

# prediction with ground truth mask
for i, pred in enumerate(pred_mask):
    gt = gt_masks[i]
    dice = (1*(pred>0.5) & gt).sum() * 2.0 / (1*(pred>0.5).sum() + gt.sum())
    print(f'Dice score for {prompts[i]}: {dice:.4f}')
    check_mask_stats(image, pred_mask[i]*255, 'X-Ray-Chest', text_prompt[i])
    print(f'p-value for {prompts[i]}: {p_value:.4f}')
```

### Usage and License Notices
The model described in this repository is provided for research and development use only. The model is not intended for use in clinical decision-making or for any other clinical use, and the performance of the model for clinical use has not been established. You bear sole responsibility for any use of this model, including incorporation into any product intended for clinical use.

### Citation
Please cite our paper if you use the code, model, or data.

Zhao, T., Gu, Y., Yang, J. et al. A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities. Nat Methods (2024). https://doi.org/10.1038/s41592-024-02499-w

```
@article{zhao2024biomedparse,
  title = {A foundation model for joint segmentation, detection, and recognition of biomedical objects across nine modalities},
  author = {Zhao, Theodore and Gu, Yu and Yang, Jianwei and Usuyama, Naoto and Lee, Ho Hin and Kiblawi, Sid and Naumann, Tristan and Gao, Jianfeng and Crabtree, Angela and Abel, Jacob and Moung-Wen, Christine and Piening, Brian and Bifulco, Carlo and Wei, Mu and Poon, Hoifung and Wang, Sheng},
  journal = {Nature Methods},
  year = {2024},
  publisher = {Nature Publishing Group UK London},
  url = {https://www.nature.com/articles/s41592-024-02499-w},
  doi = {10.1038/s41592-024-02499-w}
}
```

### Model Architecture
BiomedParse is built upon a transformer-based architecture, optimized for processing large biomedical corpora. Leveraging multi-head attention mechanisms, it excels at identifying and understanding biomedical terminology, as well as extracting contextually relevant information from dense scientific texts. The model is pre-trained on vast biomedical datasets, allowing it to generalize across various biomedical domains with high accuracy.

### Evaluation Results
Please see the paper for detailed information about methods and results. https://microsoft.github.io/BiomedParse/assets/BiomedParse_arxiv.pdf

### Fairness evaluation
We conducted fairness evaluation for different sex and age groups. Two-sided independent t-test shows non-significant differences between female and male and between different age groups, with p-value > 5% for all imaging modalities and segmentation targets evaluated.

### Ethical Considerations and Limitations
Microsoft believes Responsible AI is a shared responsibility and we have identified six principles and practices to help organizations address risks, innovate, and create value: fairness, reliability and safety, privacy and security, inclusiveness, transparency, and accountability. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant use case and addresses unforeseen product misuse. 

While testing the model with images and/or text, ensure that the data is PHI free and that there are no patient information or information that can be tracked to a patient identity.

The model is not designed for the following use cases:

- Use by clinicians to inform clinical decision-making, as a diagnostic tool or as a medical device - Although MedImageParse is highly accurate in parsing biomedical data, it is not desgined or intended to be deployed in clinical settings as-is not is it for use in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions (including to support clinical decision-making), or as a substitute of professional medical advice, diagnosis, treatment, or clinical judgment of a healthcare professional. 

- Scenarios without consent for data - Any scenario that uses health data for a purpose for which consent was not obtained.  

- Use outside of health scenarios - Any scenario that uses non-medical related image and/or serving purposes outside of the healthcare domain. 

Please see Microsoft's Responsible AI Principles and approach available at https://www.microsoft.com/en-us/ai/principles-and-approach/

### Data Specification for Deployment
- The model expect 2D 8-bit RGB or grayscale images by default, with pixel values ranging from 0 to 255 and resolution 1024*1024.
- The model outputs pixel probabilities in the same shape as the input image. We convert the floating point probabilities to 8-bit grayscale outputs. The probability threshold for segmentation mask is 0.5, which corresponds to 127.5 in 8-bit grayscale output.
- The model takes in text prompts for segmentation and doesn't have a fixed number of targets to handle. However, to ensure quality performance, we recommend the following tasks based on evaluation results. However, as we only evaluated the model on the test split of BiomedParseData, there is no guarantee for the same performance on external datasets even for the same task, due to variation in device, preprocessing, resolution and other distribution shifts. For best performance, we recommend finetuning on your specific tasks.
  - CT:
    - abdomen: adrenal gland, aorta, bladder, duodenum, esophagus, gallbladder, kidney, kidney cyst, kidney tumor, left adrenal gland, left kidney, liver, pancreas, postcava, right adrenal gland, right kidney, spleen, stomach, tumor
    - colon: tumor
    - liver: liver, tumor
    - lung: COVID-19 infection, nodule
    - pelvis: uterus
  - MRI-FLAIR: brain: edema, lower-grade glioma, tumor, tumor core, whole tumor
  - MRI-T1-Gd: brain: enhancing tumor, tumor core
  - MRI-T2: prostate: prostate peripheral zone, prostate transitional zone,
  - MRI:
    - abdomen: aorta, esophagus, gallbladder, kidney, left kidney, liver, pancreas, postcava, right kidney, spleen, stomach
    - brain: anterior hippocampus, posterior hippocampus
    - heart: left heart atrium, left heart ventricle, myocardium, right heart ventricle
    - prostate: prostate
  - OCT: retinal: edema
  - X-Ray: chest: COVID-19 infection, left lung, lung, lung opacity, right lung, viral pneumonia
  - Dermoscopy: skin: lesion, melanoma
  - Endoscope: colon: neoplastic polyp, non-neoplastic polyp, polyp
  - Fundus: retinal: optic cup, optic disc,
  - Pathology:
    - bladder: neoplastic cells
    - breast: epithelial cells, neoplastic cells
    - cervix: neoplastic cells
    - colon: glandular structure, neoplastic cells
    - esophagus: neoplastic cells
    - kidney: neoplastic cells
    - liver: epithelial cells, neoplastic cells
    - ovarian: epithelial cells, neoplastic cells
    - prostate: neoplastic cells skin: neoplastic cells
    - stomach: neoplastic cells
    - testis: epithelial cells
    - thyroid: epithelial cells, neoplastic cells
    - uterus: neoplastic cells
  - Ultrasound:
    - breast: benign tumor, malignant tumor, tumor
    - heart: left heart atrium, left heart ventricle
    - transperineal: fetal head, public symphysis