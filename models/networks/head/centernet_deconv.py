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
import math

import torch.nn as nn
from alfred.utils.log import logger

"""
normal deconv without dcn

"""


class CenternetDeconv(nn.Module):
    def __init__(self, cfg):
        super(CenternetDeconv, self).__init__()
        self.bn_momentum = cfg.MODEL.CENTERNET.BN_MOMENTUM
        # output of backbone should be [1, 2048, 16, 16]
        self.inplanes = 2048
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 256, 256],
            num_kernels=[4, 4, 4],
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding
        
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        return x



# class CenternetDeconv2(nn.Module):
#     """
#     The head used in CenterNet for object classification and box regression.
#     It has three subnet, with a common structure but separate parameters.
#     """
#     def __init__(self, cfg):
#         super(CenternetDeconv, self).__init__()
#         # modify into config
#         channels = cfg.MODEL.CENTERNET.DECONV_CHANNEL
#         deconv_kernel = cfg.MODEL.CENTERNET.DECONV_KERNEL
#         self.deconv1 = DeconvLayer(
#             channels[0], channels[1],
#             deconv_kernel=deconv_kernel[0],
#         )
#         self.deconv2 = DeconvLayer(
#             channels[1], channels[2],
#             deconv_kernel=deconv_kernel[1],
#         )
#         self.deconv3 = DeconvLayer(
#             channels[2], channels[3],
#             deconv_kernel=deconv_kernel[2],
#         )

#     def forward(self, x):
#         x = self.deconv1(x)
#         x = self.deconv2(x)
#         x = self.deconv3(x)
#         return x
