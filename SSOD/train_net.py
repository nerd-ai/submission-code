#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
# 清除之前的注册（如果有的话）
DatasetCatalog.remove("brats21_cut_train") if "brats21_cut_train" in DatasetCatalog else None
DatasetCatalog.remove("brats21_cut_val") if "brats21_cut_val" in DatasetCatalog else None

# 重新注册数据集
register_coco_instances(
    "brats21_cut_train",
    {},
    "/path/to/instance_train_pseduo_labeled.json",
    "/path/to/train/images"
)

register_coco_instances(
    "brats21_cut_val",
    {},
    "/path/to/instance_val.json",
    "/path/to/val/images"
)

# register_coco_instances(
#     "brats21_cut_train",
#     {},
#     "/root/autodl-tmp/unbiased-teacher/datasets/brats21_double_cut/annotations/instance_train.json",
#     "/root/autodl-tmp/unbiased-teacher/datasets/brats21_double_cut/train_"
# )

# register_coco_instances(
#     "brats21_cut_val",
#     {},
#     "/root/autodl-tmp/unbiased-teacher/datasets/brats21_double_cut/annotations/instance_val.json",
#     "/root/autodl-tmp/unbiased-teacher/datasets/brats21_double_cut/val_"
# )



# 训练集设置
MetadataCatalog.get("brats21_cut_train").set(thing_classes=["tumor"])
MetadataCatalog.get("brats21_cut_train").thing_dataset_id_to_contiguous_id = {1: 0}

# 验证集设置
MetadataCatalog.get("brats21_cut_val").set(thing_classes=["tumor"])
MetadataCatalog.get("brats21_cut_val").thing_dataset_id_to_contiguous_id = {1: 0}
# # 注册训练集
# register_coco_instances(
#     "brats21_cut_train",  # 数据集名称
#     {},  # 元数据（可以为空，稍后设置）
#     "/root/autodl-tmp/unbiased-teacher/datasets/brats21_cut/annotations/instances_train.json",  # 标注文件路径
#     "/root/autodl-tmp/unbiased-teacher/datasets/brats21_cut/train"  # 图像目录路径
# )

# # 注册验证集
# register_coco_instances(
#     "brats21_cut_val",  # 数据集名称
#     {},  # 元数据（可以为空，稍后设置）
#     "/root/autodl-tmp/unbiased-teacher/datasets/brats21_cut/annotations/instances_val.json",  # 标注文件路径
#     "/root/autodl-tmp/unbiased-teacher/datasets/brats21_cut/val"  # 图像目录路径
# )
# print("Datasets registered successfully!")
# # from detectron2.data import MetadataCatalog

# # 设置元数据
# MetadataCatalog.get("brats21_cut_train").set(thing_classes=["tumor"])
# MetadataCatalog.get("brats21_cut_val").set(thing_classes=["tumor"])

# # 手动设置 thing_dataset_id_to_contiguous_id
# MetadataCatalog.get("brats21_cut_train").thing_dataset_id_to_contiguous_id = {1: 0}  # 将类别 ID 1 映射到 0
# MetadataCatalog.get("brats21_cut_val").thing_dataset_id_to_contiguous_id = {1: 0}  # 将类别 ID 1 映射到 0


# # 获取元数据
# metadata_train = MetadataCatalog.get("brats21_cut_train")
# metadata_val = MetadataCatalog.get("brats21_cut_val")
# from detectron2.data import DatasetCatalog

# # 检查已注册的数据集
# print("Registered datasets:", DatasetCatalog.list())



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
