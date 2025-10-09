# 🧠 PPBoost: Weak-to-Strong Medical Segmentation Pipeline

A **modular**, **reproducible**, and **multi-dataset** framework for weakly supervised medical image segmentation.

---

## 🌐 Overview

**Modular pipeline**
```
Detection → BBox Refinement → Visual-Prompted Segmentation → Self-Training
```

**Key Features**
- 🧩 **Modular design** — easy integration or replacement of detection, refinement, or segmentation modules  
- 🧬 **Multi-dataset ready** — supports BraTS21 (MRI), LiTS17 (CT), ultrasound (kidney, etc.)  
- ⚙️ **Reproducible configs** — managed via OmegaConf/YAML with deterministic seeds  
- 📦 **Exportable artifacts** — pseudo-bboxes, SAM masks, and refined labels  
- 🧑‍🏫 **Plug-in teachers/students** — compatible with [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher)

---

## 🧰 Environment Setup

To prevent dependency and CUDA version conflicts, we recommend using **separate environments** for each module.

| Module | Framework | Reference |
|:-------|:-----------|:-----------|
| **Text-Prompted Environment** | [BiomedParse](https://github.com/microsoft/BiomedParse) | For pseudo-bbox generation |
| **Visual-Prompted Environment** | [MedSAM](https://github.com/bowang-lab/MedSAM) | For SAM-based segmentation |
| **SSOD Environment** | [Unbiased-Teacher](https://github.com/facebookresearch/unbiased-teacher) | For semi-supervised object detection |

---

## 🗂️ Datasets

We will provide hosted copies of **BraTS21**, **LiTS17**, and **Ultrasound (Kidney)** datasets via a shared **Google Drive link** soon.

> ⚠️ Note: Datasets are not included in the repository.  
> Please place them externally and update the configuration paths accordingly.

---

## 🚀 Training Pipeline

### 1️⃣ Text-to-Pseudo-BBox Generation

Generate confidence maps at two temperatures:

```bash
python BiomedParse/inference_uncertain_random.py
```
*(Set different `temperature_new` parameters)*

Compute KL divergence between the two confidence maps:
```bash
python src/filtering/kl_calculation.py
```

Generate pseudo-bbox annotations using temperature=1 confidence maps:
```bash
python src/filtering/pseudo_bbox_annotation.py
```

Filter and select reliable image filenames:
```bash
python src/filtering/filter.py
```

---

### 2️⃣ Pseudo-BBox-Based SSOD (Semi-Supervised Object Detection)

Convert the reliable filenames to corresponding image IDs in COCO format:
```bash
python SSOD/coco_data_construction/convert_to_ids.py
```

Assign supervision pseudo-labels:
```bash
python SSOD/coco_data_construction/supervison_id_assign.py
```

Generate training, validation, and test COCO JSON files:
```bash
python SSOD/coco_data_construction/generate_coco_format.py
```

Train the SSOD model:
```bash
bash SSOD/run_train.sh
```

---

## 🧪 Inference Pipeline

### 1️⃣ Object Detection for Test Data
```bash
bash SSOD/run_test.sh
```

Convert the inference COCO JSON file for MedSAM prompting:
```bash
python src/convert_to_medsam_infer.py
```

---

### 2️⃣ Pseudo-BBox Expansion
```bash
python src/expansion/bbox_expand_by_score.py
```

---

### 3️⃣ Visual-Prompted Segmentation (MedSAM)
```bash
bash MedSAM/batch_medsam_inference.py
```

---

### 4️⃣ Evaluation Metrics
```bash
bash src/evaluation/eval.bash
```

---
