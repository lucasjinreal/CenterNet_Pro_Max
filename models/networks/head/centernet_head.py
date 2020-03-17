#!/usr/bin/python3
#
# Copyright (c) 2020 jintian.
#
# This file is part of centernet_pro
# (see jinfgang.github.io).
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
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class SingleHead(nn.Module):

    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class CenternetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self, cfg):
        super(CenternetHead, self).__init__()
        self.cfg = cfg
        self._init_heads()

    def _init_heads(self):
        if self.cfg.MODEL.CENTERNET.USE_DCN:
            self.cls_head = SingleHead(
                64,
                self.cfg.MODEL.CENTERNET.NUM_CLASSES,
                bias_fill=True,
                bias_value=self.cfg.MODEL.CENTERNET.BIAS_VALUE,
            )
            self.wh_head = SingleHead(64, 2)
            self.reg_head = SingleHead(64, 2)
        else:
            # build heads for resnets
            num_output = self.cfg.MODEL.CENTERNET.NUM_CLASSES
            self.cls_head = nn.Sequential(
                nn.Conv2d(256, self.cfg.MODEL.CENTERNET.HEAD_CONV,
                          kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cfg.MODEL.CENTERNET.HEAD_CONV, num_output,
                          kernel_size=1, stride=1, padding=0))
            self.wh_head = SingleHead(256, 2)
            self.reg_head = SingleHead(256, 2)

    def forward(self, x):
        cls = self.cls_head(x)
        cls = torch.sigmoid(cls)
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        pred = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        return pred
