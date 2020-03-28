#!/usr/bin/python3
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

import torch.nn as nn
import torchvision.models.resnet as resnet

from .backbone import Backbone

_resnet_mapper = {
    18: resnet.resnet18,
    50: resnet.resnet50,
    101: resnet.resnet101,
}


class ResnetBackbone(Backbone):

    def __init__(self, cfg, input_shape=None, pretrained=True):
        super().__init__()
        depth = cfg.MODEL.RESNETS.DEPTH
        backbone = _resnet_mapper[depth](pretrained=pretrained)
        self.stage0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
