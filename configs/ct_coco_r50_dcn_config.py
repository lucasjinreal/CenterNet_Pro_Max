"""

custom your config here

"""
from models.configs.base_detection_config import config, BaseDetectionConfig
import os.path as osp


_config_dict = dict(
    MODEL=dict(
        # WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-18.pth",
        WEIGHTS="",
        MASK_ON=False,
        RESNETS=dict(DEPTH=50),
        PIXEL_MEAN=[0.485, 0.456, 0.406],
        PIXEL_STD=[0.229, 0.224, 0.225],
        CENTERNET=dict(
            USE_DCN=True,
            HEAD_CONV=64,
            DECONV_CHANNEL=[2048, 256, 128, 64],
            DECONV_KERNEL=[4, 4, 4],
            NUM_CLASSES=80,
            MODULATE_DEFORM=True,
            BIAS_VALUE=-2.19,
            DOWN_SCALE=4,
            MIN_OVERLAP=0.7,
            TENSOR_DIM=128,
            BN_MOMENTUM=0.1
        ),
        LOSS=dict(
            CLS_WEIGHT=1,
            WH_WEIGHT=0.1,
            REG_WEIGHT=1,
        ),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ('CenterAffine', dict(
                    boarder=128,
                    output_size=(512, 512),
                    random_aug=True)),
                ('RandomFlip', dict()),
                ('RandomBrightness', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomContrast', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomSaturation', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomLighting', dict(scale=0.1)),
            ],
            TEST_PIPELINES=[
            ],
        ),
        FORMAT="RGB",
        OUTPUT_SIZE=(128, 128),
        MIN_SIZE_TEST=512,
        MAX_SIZE_TEST=512,
    ),
    DATALOADER=dict(
        NUM_WORKERS=0,
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        OPTIMIZER=dict(
            NAME="Adam",
            BASE_LR=0.0125,
            WEIGHT_DECAY=1e-3,
            AMSGRAD=True,
        ),
        LR_SCHEDULER=dict(
            GAMMA=0.1,
            STEPS=(81000, 108000),
            MAX_ITER=126000,
            WARMUP_ITERS=1000,
        ),
        IMS_PER_BATCH=8,
    ),
    OUTPUT_DIR='./checkpoints/',
    GLOBAL=dict(DUMP_TEST=False),
    HOOKS=dict(
        EVAL_PERIOD=500,
        LOG_PERIOD=50,
    )
)


class CenterNetConfig(BaseDetectionConfig):
    def __init__(self):
        super(CenterNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CenterNetConfig()