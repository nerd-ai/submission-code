import os
import argparse
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import transform
from tqdm import tqdm

from MedSAM.segment_anything import sam_model_registry

class MedSAMPredictor:
    def __init__(self, checkpoint_path, gpu_ids=None):
        """
        Initialize MedSAM predictor.
        
        Args:
            checkpoint_path: MedSAM 模型 checkpoint 的路径。
            gpu_ids: 要使用的 GPU ID 列表；如果为 None，则自动选取所有可用的 GPU。
        """
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        assert len(gpu_ids) >= 1, "No CUDA devices available."
        self.gpu_ids = gpu_ids
        self.device = torch.device(f'cuda:{gpu_ids[0]}')

        torch.backends.cudnn.benchmark = True
        # 允许使用 TF32 以加速 Ampere+ GPU 上的卷积/矩阵计算（对精度影响极小）
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        # 加载模型到当前进程对应的单个 GPU（多 GPU 通过多进程分片实现）
        self.sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.sam_model.to(self.device)
        self.sam_model.eval()
        self.medsam_model = self.sam_model

    def get_image_embedding(self, image_path):
        """获取单张图像的 embedding，预处理并统一缩放至 1024×1024"""
        # 读取图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        else:
            if isinstance(image_path, torch.Tensor):
                image = image_path.cpu().numpy()
            else:
                image = image_path
                
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

        H, W = image.shape[:2]
        
        # 缩放和归一化
        img_1024 = transform.resize(
            image, (1024, 1024), 
            order=3, 
            preserve_range=True, 
            anti_aliasing=True
        ).astype(np.float32)
        
        img_min = img_1024.min()
        img_range = np.maximum(img_1024.max() - img_min, 1e-8)
        img_1024 = (img_1024 - img_min) / img_range
        
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.medsam_model.image_encoder(img_1024_tensor)
        
        return image_embedding, (H, W)
    
    def get_batch_image_embeddings(self, image_paths: List[str]):
        """
        对一批图像进行预处理并提取图像 embedding。
        
        Args:
            image_paths: 图像路径列表。
        
        Returns:
            img_embeds: 图像 embedding 的批次，形状为 (N, C, H_feat, W_feat)。
            sizes: 每张图像的原始尺寸列表 [(H, W), ...]。
        """
        imgs: List[torch.Tensor] = []
        sizes: List[Tuple[int, int]] = []
        for image_path in image_paths:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
            else:
                if isinstance(image_path, torch.Tensor):
                    image = image_path.cpu().numpy()
                else:
                    image = image_path

            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            H, W = image.shape[:2]
            sizes.append((H, W))
            img_1024 = transform.resize(
                image, (1024, 1024),
                order=3,
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.float32)
            img_min = img_1024.min()
            img_range = np.maximum(img_1024.max() - img_min, 1e-8)
            img_1024 = (img_1024 - img_min) / img_range
            img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1)
            imgs.append(img_tensor)
        imgs_tensor = torch.stack(imgs, dim=0).to(self.device)
        with torch.inference_mode():
            img_embeds = self.medsam_model.image_encoder(imgs_tensor)
        return img_embeds, sizes

    def predict_single_image(self, img_embed, box, size):
        """单张图像的 mask 预测"""
        H, W = size
        # 将 bbox 从原始尺寸映射到 1024 尺度
        box_1024 = np.ceil(box / np.array([W, H, W, H]) * 1024).astype(int)
        
        if len(box_1024.shape) == 1:
            box_1024 = box_1024[None, :]
        
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)
        
        with torch.inference_mode():
            sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            
            low_res_logits, _ = self.medsam_model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            low_res_pred = torch.sigmoid(low_res_logits)
            pred_mask = F.interpolate(
                low_res_pred,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_mask = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            
            return pred_mask

    def predict_batch(self, img_embeds, bboxes, sizes):
        """
        对一批图像同时进行 mask 预测。
        
        Args:
            img_embeds: 图像 embedding 的 Tensor，形状 (N, C, H_feat, W_feat)。
            bboxes: 每张图像的 bbox 列表（原始图像坐标），格式为 [x1, y1, x2, y2]。
            sizes: 每张图像的原始尺寸列表 [(H, W), ...]。
        
        Returns:
            pred_masks: 每张图像预测的二值 mask 列表。
        """
        # 将每张图像的 bbox 从原始尺寸映射到 1024 尺度
        scaled_boxes = []
        for bbox, (H, W) in zip(bboxes, sizes):
            scaled_box = np.ceil(bbox / np.array([W, H, W, H]) * 1024).astype(int)
            scaled_boxes.append(scaled_box)
        scaled_boxes = np.stack(scaled_boxes, axis=0)
        box_torch = torch.as_tensor(scaled_boxes, dtype=torch.float, device=self.device)
        
        with torch.inference_mode():
            sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            
            low_res_logits, _ = self.medsam_model.mask_decoder(
                image_embeddings=img_embeds,
                image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            low_res_pred = torch.sigmoid(low_res_logits)
            pred_masks = []
            # 对每个样本分别上采样到其原始尺寸
            for i, (H, W) in enumerate(sizes):
                mask = F.interpolate(
                    low_res_pred[i:i+1],
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                mask = (mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                pred_masks.append(mask)
            
            return pred_masks

    def get_all_images(self, directory):
        """获取目录下所有图像的文件路径"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.nii', '.nii.gz')
        image_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        return sorted(image_files)

    def process_dataset(self, image_dir, bbox_json_path, output_dir, batch_size=4):
        """
        使用 JSON 文件中的 bbox 对目录下的所有图像进行预测，并保存预测的 mask。
        使用批处理方式一次同时处理多张图像。
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取 bbox 的 JSON 文件
        with open(bbox_json_path, 'r') as f:
            bbox_data = json.load(f)
        
        # 建立文件名到 bbox 的映射
        filename_to_bbox = {}
        for item in bbox_data:  # 假设 bbox_data 为列表形式
            x1, y1, w, h = item['bbox']
            x2 = x1 + w
            y2 = y1 + h
            filename_to_bbox[item['file_name']] = {
                'bbox': [x1, y1, x2, y2],  # 采用 x1, y1, x2, y2 格式
                'score': item['score']
            }
        
        # 获取所有图像文件
        image_files = self.get_all_images(image_dir)
        
        # 过滤出有 bbox 的图像
        valid_images = []
        valid_bboxes = []
        valid_filenames = []
        for image_path in image_files:
            image_filename = os.path.basename(image_path)
            if image_filename not in filename_to_bbox:
                print(f"Warning: No bbox found for {image_filename}")
                continue
            valid_images.append(image_path)
            valid_bboxes.append(np.array(filename_to_bbox[image_filename]['bbox']))
            valid_filenames.append(image_filename)
        
        self.process_prepared_dataset(valid_images, valid_bboxes, valid_filenames, output_dir, batch_size)
        
        print(f"\nProcessing completed:")
        print(f"Total images processed: {len(valid_images)}")
        print(f"Masks saved to: {output_dir}")

    def process_prepared_dataset(self, valid_images: List[str], valid_bboxes: List[np.ndarray], valid_filenames: List[str], output_dir: str, batch_size: int = 4):
        os.makedirs(output_dir, exist_ok=True)
        for i in tqdm(range(0, len(valid_images), batch_size), desc=f"GPU {self.device.index} processing"):
            batch_paths = valid_images[i:i + batch_size]
            batch_bboxes = valid_bboxes[i:i + batch_size]
            batch_filenames = valid_filenames[i:i + batch_size]

            img_embeds, sizes = self.get_batch_image_embeddings(batch_paths)
            pred_masks = self.predict_batch(img_embeds, batch_bboxes, sizes)

            for filename, mask in zip(batch_filenames, pred_masks):
                image_basename = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{image_basename}.png")
                mask_image = Image.fromarray(mask * 255)
                mask_image.save(output_path)


def build_dataset_lists(image_dir: str, bbox_json_path: str):
    """Build aligned lists of image paths, bboxes and file names filtered by bbox availability."""
    def get_all_images(directory):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.nii', '.nii.gz')
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
        return sorted(image_files)

    with open(bbox_json_path, 'r') as f:
        bbox_data = json.load(f)

    filename_to_bbox = {}
    for item in bbox_data:
        x1, y1, w, h = item['bbox']
        x2 = x1 + w
        y2 = y1 + h
        filename_to_bbox[item['file_name']] = {'bbox': [x1, y1, x2, y2], 'score': item['score']}

    image_files = get_all_images(image_dir)

    valid_images: List[str] = []
    valid_bboxes: List[np.ndarray] = []
    valid_filenames: List[str] = []
    for image_path in image_files:
        image_filename = os.path.basename(image_path)
        if image_filename not in filename_to_bbox:
            continue
        valid_images.append(image_path)
        valid_bboxes.append(np.array(filename_to_bbox[image_filename]['bbox']))
        valid_filenames.append(image_filename)

    return valid_images, valid_bboxes, valid_filenames


def shard_lists(valid_images, valid_bboxes, valid_filenames, num_shards, shard_id):
    """Return the shard slice for this worker."""
    assert len(valid_images) == len(valid_bboxes) == len(valid_filenames)
    n = len(valid_images)
    shard_images, shard_bboxes, shard_filenames = [], [], []
    for idx in range(shard_id, n, num_shards):
        shard_images.append(valid_images[idx])
        shard_bboxes.append(valid_bboxes[idx])
        shard_filenames.append(valid_filenames[idx])
    return shard_images, shard_bboxes, shard_filenames


def _mp_worker(local_rank, gpu_id, checkpoint_path, batch_size, out_dir,
               valid_images, valid_bboxes, valid_filenames):
    """Top-level multiprocessing worker to allow pickling under spawn."""
    shard_imgs, shard_boxes, shard_names = shard_lists(
        valid_images, valid_bboxes, valid_filenames,
        num_shards=len(set([gpu_id] + [])),  # placeholder, will be overridden in main call
        shard_id=local_rank,
    )
    # The above num_shards trick is fragile; main passes already-sharded lists. So instead, recompute properly here.
    # Recompute shards correctly using provided shard_id and overall world size passed via env.
    world_size = int(os.environ.get('MEDSAM_WORLD_SIZE', '1'))
    shard_imgs, shard_boxes, shard_names = shard_lists(
        valid_images, valid_bboxes, valid_filenames,
        num_shards=world_size,
        shard_id=local_rank,
    )
    if len(shard_imgs) == 0:
        return
    torch.cuda.set_device(gpu_id)
    predictor = MedSAMPredictor(checkpoint_path, [gpu_id])
    predictor.process_prepared_dataset(shard_imgs, shard_boxes, shard_names, out_dir, batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description="MedSAM batch inference (multi-GPU)")
    parser.add_argument("--checkpoint", type=str, default="/path/to/checkpoint/", help="Path to MedSAM checkpoint .pth")
    parser.add_argument("--images", type=str, required=True, help="Directory of images")
    parser.add_argument("--bboxes", type=str, required=True, help="Path to bbox JSON file")
    parser.add_argument("--out", type=str, required=True, help="Output directory for masks")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids, e.g., 0,1,2. Default: all available")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpus is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [int(x) for x in args.gpus.split(',') if x.strip() != '']
    assert len(gpu_ids) >= 1, "No GPUs specified or found."

    # 预构建数据列表一次，然后分片给各 GPU 处理
    valid_images, valid_bboxes, valid_filenames = build_dataset_lists(args.images, args.bboxes)

    if len(gpu_ids) == 1:
        predictor = MedSAMPredictor(args.checkpoint, gpu_ids)
        predictor.process_prepared_dataset(valid_images, valid_bboxes, valid_filenames, args.out, args.batch_size)
        print("Done on single GPU.")
        return

    # 多 GPU：每块 GPU 处理一个数据分片（独立进程）
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    processes = []
    os.environ['MEDSAM_WORLD_SIZE'] = str(len(gpu_ids))
    for local_rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(target=_mp_worker, args=(local_rank, gpu_id, args.checkpoint, args.batch_size, args.out,
                                                valid_images, valid_bboxes, valid_filenames))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("All GPU workers finished.")


if __name__ == "__main__":
    main()
