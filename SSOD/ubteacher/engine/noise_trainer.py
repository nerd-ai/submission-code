from collections import defaultdict
import numpy as np
import torch
import logging
import time
from detectron2.engine import hooks
from detectron2.utils import comm  

from ubteacher.engine.trainer import BaselineTrainer

class NoiseAnalysisTrainer(BaselineTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)
        self.image_loss_history = defaultdict(list)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        # 计算并记录每张图片的loss
        image_losses = self._compute_per_image_loss(data, record_dict)
        for idx, image_data in enumerate(data):
            image_id = image_data.get("image_id", idx)
            self.image_loss_history[image_id].append(image_losses[idx])

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _compute_per_image_loss(self, data, record_dict):
        """计算每张图片的loss"""
        per_image_losses = []
        
        # 获取所有loss components
        loss_components = {k: v for k, v in record_dict.items() 
                         if k[:4] == "loss" and k[-3:] != "val"}
        
        total_gt_boxes = sum(len(d["instances"]) for d in data)
        for image_data in data:
            num_gt_boxes = len(image_data["instances"])
            if num_gt_boxes > 0:
                # 按gt boxes数量加权计算单张图片的loss
                image_loss = sum(loss_components.values()) * (num_gt_boxes / total_gt_boxes)
                per_image_losses.append(image_loss.item())
            else:
                per_image_losses.append(0.0)
        
        return per_image_losses

    def analyze_noise(self):
        """分析loss patterns识别noisy标注"""
        window_size = self.cfg.NOISEANALYSIS.WINDOW_SIZE
        clean_images = []
        noisy_images = []
        
        # 计算每张图片的loss统计特征
        for image_id, losses in self.image_loss_history.items():
            if len(losses) < window_size:
                continue
                
            recent_losses = losses[-window_size:]
            mean_loss = np.mean(recent_losses)
            std_loss = np.std(recent_losses)
            trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            # 使用简单的阈值规则
            if mean_loss < self.cfg.NOISEANALYSIS.MEAN_THRESH and \
               std_loss < self.cfg.NOISEANALYSIS.STD_THRESH and \
               trend < 0:
                clean_images.append(image_id)
            else:
                noisy_images.append(image_id)
        
        return clean_images, noisy_images

    def build_hooks(self):
        """
        Build a list of default hooks.
        """
        # 获取原有的所有hooks
        hooks_list = super().build_hooks()
        
        # Noise analysis可以作为评估的一部分
        # 通过现有的EvalHook定期执行
        def noise_analysis_func():
            clean_images, noisy_images = self.analyze_noise()
            # 返回结果以供记录
            return {
                "noise/num_clean": len(clean_images),
                "noise/num_noisy": len(noisy_images),
                "noise/clean_ratio": len(clean_images) / (len(clean_images) + len(noisy_images))
            }
        
        if comm.is_main_process():
            # 使用EvalHook来执行noise analysis
            hooks_list.append(
                hooks.EvalHook(
                    self.cfg.NOISEANALYSIS.ANALYSIS_PERIOD,
                    noise_analysis_func
                )
            )
            
        return hooks_list
