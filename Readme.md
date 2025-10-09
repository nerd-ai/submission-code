Modular pipeline: detection → bbox refinement → visual-prompted segmentation → self-training.

Multi-dataset ready: BraTS21 (MRI), LiTS17 (CT), ultrasound (kidney, etc.).

Reproducible configs (OmegaConf/YAML) and deterministic seeds.

Exportable artifacts: pseudo-bboxes, SAM masks.

Plug-in teachers/students: Unbiased Teacher


Environment Setup

We separate environments to avoid dependency/CUDA conflicts. Install them individually depending on your module.

Text‑prompted Environment (BiomedParse): Refer to https://github.com/microsoft/BiomedParse

Visual-prompted Environment (MedSAM): Refer to https://github.com/bowang-lab/MedSAM

SSOD environment (Unbiased-Teacher): Refer to https://github.com/facebookresearch/unbiased-teacher


Datasets:
We will provide hosted copies of the datasets via a shared Google Drive link later


Training:

1. Text-to-pseudo-bbox:

Run python BiomedParse/inference_uncertain_random.py (set two different temperature_new parameters)
and then save confidence maps with two different temperatures as .npy files

Caculate the kl divergence for each samples between the confidence maps with different temperatures:
python src/filtering/kl_calculation.py

Use all pseudo bboxes from the confidence maps (temperature=1) to generate pseudo-bbox annotations:
python /home/xli263/PPBoost/src/filtering/pseudo_bbox_annotation.py

and then filter and select the reliable image filenames:
python /src/filtering/filter.py

2. pseudo-bbox based SSOD

Use the following command to convert the reliable image filenames to the corresponding image ids in the coco-format .json file:
python SSOD/coco_data_construction/convert_to_ids.py

Generate the supervision pseudo-labels:
python SSOD/coco_data_construction/supervison_id_assign.py

Generate training, validation and test coco-format json file:
python SSOD/coco_data_construction/generate_coco_format.py

Train the SSOD:
bash SSOD/run_train.sh

Inference:
1. Object detection for test data:
bash /home/xli263/PPBoost/SSOD/run_test.sh

and then convert the obtained inference coco .json file to a new .json file for prompting MedSAM:
python src/convert_to_medsam_infer.py

2. Pseudo bbox prompts expansion:
python /home/xli263/PPBoost/src/expansion/bbox_expand_by_score.py


3. Prompt MedSAM:
bash /home/xli263/PPBoost/MedSAM/batch_medsam_inference.py

4. Evaluation metrics:
bash /home/xli263/PPBoost/src/evaluation/eval.bash
