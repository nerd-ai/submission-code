from detectron2.config import CfgNode as CN

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
    _C.SEMISUPNET.BURN_UP_STEP = 2000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.9996
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 5  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 1  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/BRATS_cut_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True


    # 添加 noise analysis 相关配置
    _C.NOISEANALYSIS = CN()
    # 多少次iteration进行一次分析
    _C.NOISEANALYSIS.ANALYSIS_PERIOD = 100  
    # loss相关阈值
    _C.NOISEANALYSIS.MEAN_THRESH = 1.0
    _C.NOISEANALYSIS.STD_THRESH = 0.5
    # 最少需要多少次观察才进行分析
    _C.NOISEANALYSIS.MIN_OBSERVATIONS = 3