# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


# def add_ubteacher_config(cfg):
#     """
#     Add config for semisupnet.
#     """
#     _C = cfg
#     _C.TEST.VAL_LOSS = True

#     _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
#     _C.MODEL.RPN.LOSS = "CrossEntropy"
#     _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

#     _C.SOLVER.IMG_PER_BATCH_LABEL = 1
#     _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
#     _C.SOLVER.FACTOR_LIST = (1,)

#     _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
#     _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
#     _C.DATASETS.CROSS_DATASET = False
#     _C.TEST.EVALUATOR = "COCOeval"

#     _C.SEMISUPNET = CN()

#     # Output dimension of the MLP projector after `res5` block
#     _C.SEMISUPNET.MLP_DIM = 128

#     # Semi-supervised training
#     _C.SEMISUPNET.Trainer = "ubteacher"
#     _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
#     _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
#     _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
#     _C.SEMISUPNET.BURN_UP_STEP = 12000
#     _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
#     _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
#     _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
#     _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"

#     # dataloader
#     # supervision level
#     _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
#     _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
#     _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

#     _C.EMAMODEL = CN()
#     _C.EMAMODEL.SUP_CONSIST = True

# def add_ubteacher_config(cfg):
#     _C = cfg
    
#     # 修改数据集相关配置
#     _C.DATASETS.TRAIN_LABEL = ("brats21_train",)  # 改为BraTS数据集名称
#     _C.DATASETS.TRAIN_UNLABEL = ("brats21_train",)  # 使用同一数据集
#     _C.DATASETS.CROSS_DATASET = False  # 保持False，因为我们使用同一个数据集
#     _C.TEST.EVALUATOR = "COCOeval"  # 可以保持使用COCO评估器，因为我们使用了COCO格式
    
#     # 调整半监督学习参数
#     _C.SEMISUPNET.BBOX_THRESHOLD = 0.7  # 可以保持这个置信度阈值
#     _C.SEMISUPNET.BURN_UP_STEP = 2000   # 由于是医学图像，可以减少burn-up步数
#     _C.SEMISUPNET.EMA_KEEP_RATE = 0.9996  # teacher模型更新率
#     _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0  # 无监督损失权重
    
#     # 数据加载器设置
#     _C.DATALOADER.SUP_PERCENT = 5.0  # 使用5%的数据作为标注数据
#     _C.DATALOADER.RANDOM_DATA_SEED = 1  # 随机种子
#     # 修改为BraTS的种子文件路径，或者我们可以直接在代码中处理，不使用文件
#     _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/BRATS_supervision.txt"
    
#     # 批次大小设置
#     _C.SOLVER.IMG_PER_BATCH_LABEL = 8    # 根据GPU显存调整
#     _C.SOLVER.IMG_PER_BATCH_UNLABEL = 8  # 通常与label批次大小相同

# def add_ubteacher_config(cfg):

#     _C = cfg
#     _C.SEMISUPNET = CN()
#     _C.TEST.VAL_LOSS = True

#     _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
#     _C.MODEL.RPN.LOSS = "CrossEntropy"
#     _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

#     _C.SOLVER.IMG_PER_BATCH_LABEL = 1
#     _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
#     _C.SOLVER.FACTOR_LIST = (1,)
#     # 修改数据集相关配置
#     _C.DATASETS.TRAIN_LABEL = ("brats21_train",)  # 改为BraTS数据集名称
#     _C.DATASETS.TRAIN_UNLABEL = ("brats21_train",)  # 使用同一数据集
#     _C.DATASETS.CROSS_DATASET = False  # 保持False，因为我们使用同一个数据集
#     _C.TEST.EVALUATOR = "COCOeval"  # 可以保持使用COCO评估器，因为我们使用了COCO格式
    
#     # 调整半监督学习参数
#     _C.SEMISUPNET.BBOX_THRESHOLD = 0.7  # 可以保持这个置信度阈值
#     _C.SEMISUPNET.BURN_UP_STEP = 2000   # 由于是医学图像，可以减少burn-up步数
#     _C.SEMISUPNET.EMA_KEEP_RATE = 0.9996  # teacher模型更新率
#     _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0  # 无监督损失权重
    
#     # 数据加载器设置
#     _C.DATALOADER.SUP_PERCENT = 5.0  # 使用5%的数据作为标注数据
#     _C.DATALOADER.RANDOM_DATA_SEED = 1  # 随机种子
#     # 修改为BraTS的种子文件路径，或者我们可以直接在代码中处理，不使用文件
#     _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/BRATS_supervision.txt"
    
#     # # 批次大小设置
#     # _C.SOLVER.IMG_PER_BATCH_LABEL = 8    # 根据GPU显存调整
#     # _C.SOLVER.IMG_PER_BATCH_UNLABEL = 8  # 通常与label批次大小相同

# def add_ubteacher_config(cfg):
#     """
#     Add config for semisupnet.
#     """
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

#     _C = cfg
#     _C.TEST.VAL_LOSS = True

#     _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
#     _C.MODEL.RPN.LOSS = "CrossEntropy"
#     _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

#     _C.SOLVER.IMG_PER_BATCH_LABEL = 32
#     _C.SOLVER.IMG_PER_BATCH_UNLABEL = 32
#     _C.SOLVER.FACTOR_LIST = (1,)

#     _C.DATASETS.TRAIN_LABEL = ("brats21_cut_train",)
#     _C.DATASETS.TRAIN_UNLABEL = ("brats21_cut_train",)
#     _C.DATASETS.CROSS_DATASET = False
#     _C.TEST.EVALUATOR = "COCOeval"

#     _C.SEMISUPNET = CN()

#     # Output dimension of the MLP projector after `res5` block
#     _C.SEMISUPNET.MLP_DIM = 128

#     # Semi-supervised training
#     _C.SEMISUPNET.Trainer = "ubteacher"
#     _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
#     _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
#     _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
#     _C.SEMISUPNET.BURN_UP_STEP = 2000
#     _C.SEMISUPNET.EMA_KEEP_RATE = 0.9996
#     _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
#     _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
#     _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"

#     # dataloader
#     # supervision level
#     _C.DATALOADER.SUP_PERCENT = 5  # 5 = 5% dataset as labeled set
#     _C.DATALOADER.RANDOM_DATA_SEED = 1  # random seed to read data
#     _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/BRATS_cut_supervision.txt"

#     _C.EMAMODEL = CN()
#     _C.EMAMODEL.SUP_CONSIST = True



def add_ubteacher_config(cfg):
    """
    Add config for semisupnet.
    """
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 32
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 32
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("brats21_cut_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("brats21_cut_train",)
    _C.DATASETS.CROSS_DATASET = False
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "ubteacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 300
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.9996
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 10  # 10 = 10% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 3  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "/path/to/supervision.txt"  # set your path here

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True
