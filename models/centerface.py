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
import math

import numpy as np
import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

# disable fancy structures
from .ops.shape_spec import ShapeSpec

from alfred.utils.log import logger as logging

from .networks.head.centerface_head import CenterFaceDeconv
from .backbone.backbone import Backbone
from .backbone.mobilenet_backbone import MobileNetV2, model_urls


class CenterFace(nn.Module):
    def __init__(self, base_name, heads, head_conv=24, pretrained=True):
        super(CenterFace, self).__init__()
        self.heads = heads
        self.base = globals()[base_name](
            pretrained=pretrained)
        channels = self.base.feat_channel
        self.dla_up = CenterFaceDeconv(channels, out_dim=head_conv)
        for head in self.heads:
            classes = self.heads[head]
            if head == 'hm':
                fc = nn.Sequential(
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True),
                    nn.Sigmoid()
                )
            else:
                fc = nn.Conv2d(head_conv, classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
            # if 'hm' in head:
            #     fc.bias.data.fill_(-2.19)
            # else:
            #     nn.init.normal_(fc.weight, std=0.001)
            #     nn.init.constant_(fc.bias, 0)
            self.__setattr__(head, fc)

    # @dict2list         # 转onnx的时候需要将输出由dict转成list模式
    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def load_model(model, state_dict):
    new_model = model.state_dict()
    new_keys = list(new_model.keys())
    old_keys = list(state_dict.keys())
    restore_dict = OrderedDict()
    for id in range(len(new_keys)):
        restore_dict[new_keys[id]] = state_dict[old_keys[id]]
    model.load_state_dict(restore_dict)


def mobilenetv2_10(pretrained=True, **kwargs):
    model = MobileNetV2(width_mult=1.0)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mobilenet_v2'],
                                        progress=True)
        load_model(model, state_dict)
    return model


def mobilenetv2_5(pretrained=False, **kwargs):
    model = MobileNetV2(width_mult=0.5)
    if pretrained:
        print('This version does not have pretrain weights.')
    return model


# num_layers  : [10 , 5]
def build_model(num_layers, heads, head_conv=24):
    model = CenterFace('mobilenetv2_{}'.format(num_layers), heads,
                       pretrained=True,
                       head_conv=head_conv)
    return model


if __name__ == '__main__':
    inp = torch.zeros([1, 3, 416, 416])
    model = build_model(10, {'hm': 1, 'hm_offset': 2, 'wh': 2, 'landmarks': 10},
                        head_conv=24)  # hm reference for the classes of objects//这个头文件只能做矩形框检测
    res = model(inp)
    print(res.shape)
