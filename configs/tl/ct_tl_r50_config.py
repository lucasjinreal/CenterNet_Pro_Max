<<<<<<< HEAD
#
# Copyright (c) 2020 jintian.
#
# This file is part of CenterNet_Pro_Max
# (see jinfagang.github.io).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
"""

custom your config here


this is used for training trafficlight detection model


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
            USE_DCN=False,
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
        # this lr work on 1 gpu, try SGD as well
        OPTIMIZER=dict(
            NAME="Adam",
            BASE_LR=1.25e-3,
            WEIGHT_DECAY=1e-4,
            AMSGRAD=True,
        ),
        LR_SCHEDULER=dict(
            GAMMA=0.1,
            STEPS=(241000, 408000),
            MAX_ITER=826000,
            WARMUP_ITERS=2000,
        ),
        IMS_PER_BATCH=12,
    ),
    OUTPUT_DIR='./checkpoints/',
    GLOBAL=dict(DUMP_TEST=False),
    HOOKS=dict(
        # judge this by epochs rather than iters, different dataset has different iters
        EVAL_PERIOD=500,
        LOG_PERIOD=50,
    )
)


class CenterNetConfig(BaseDetectionConfig):
    def __init__(self):
        super(CenterNetConfig, self).__init__()
        self._register_configuration(_config_dict)


=======
#
# Copyright (c) 2020 jintian.
#
# This file is part of CenterNet_Pro_Max
# (see jinfagang.github.io).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
"""

custom your config here


this is used for training trafficlight detection model


"""
from models.configs.base_detection_config import config, BaseDetectionConfig
import os.path as osp


"""
4 classes on trafficlight

"""

_config_dict = dict(
    MODEL=dict(
        # WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-18.pth",
        WEIGHTS="",
        MASK_ON=False,
        RESNETS=dict(DEPTH=50),
        PIXEL_MEAN=[0.485, 0.456, 0.406],
        PIXEL_STD=[0.229, 0.224, 0.225],
        CENTERNET=dict(
            USE_DCN=False,
            HEAD_CONV=64,
            DECONV_CHANNEL=[2048, 256, 128, 64],
            DECONV_KERNEL=[4, 4, 4],
            NUM_CLASSES=4,
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
        TRAIN=("coco_tl",),
        # TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        # this lr work on 1 gpu, try SGD as well
        OPTIMIZER=dict(
            NAME="Adam",
            BASE_LR=1.25e-3,
            WEIGHT_DECAY=1e-4,
            AMSGRAD=True,
        ),
        LR_SCHEDULER=dict(
            GAMMA=0.1,
            STEPS=(241000, 408000),
            MAX_ITER=826000,
            WARMUP_ITERS=2000,
        ),
        IMS_PER_BATCH=12,
    ),
    OUTPUT_DIR='./checkpoints/',
    GLOBAL=dict(DUMP_TEST=False),
    HOOKS=dict(
        # judge this by epochs rather than iters, different dataset has different iters
        EVAL_PERIOD=500,
        LOG_PERIOD=50,
    )
)


class CenterNetConfig(BaseDetectionConfig):
    def __init__(self):
        super(CenterNetConfig, self).__init__()
        self._register_configuration(_config_dict)


>>>>>>> 3110c98084fd2aaf2784db21de83a269f099d90c
config = CenterNetConfig()