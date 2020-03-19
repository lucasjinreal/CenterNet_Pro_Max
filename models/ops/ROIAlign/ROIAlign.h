/*
 * Copyright (c) 2020 jintian.
 *
 * This file is part of CenterNet_Pro_Max
 * (see jinfagang.github.io).
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/types.h>

namespace dl_lib {

at::Tensor ROIAlign_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    bool aligned);

at::Tensor ROIAlign_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    bool aligned);

#ifdef WITH_CUDA
at::Tensor ROIAlign_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    bool aligned);

at::Tensor ROIAlign_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    bool aligned);
#endif

// Interface for Python
inline at::Tensor ROIAlign_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    bool aligned) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_forward_cuda(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        aligned);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIAlign_forward_cpu(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

inline at::Tensor ROIAlign_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    bool aligned) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_backward_cuda(
        grad,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width,
        sampling_ratio,
        aligned);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIAlign_backward_cpu(
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width,
      sampling_ratio,
      aligned);
}

} // namespace dl_lib
