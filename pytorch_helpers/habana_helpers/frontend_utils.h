/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#pragma once

#include <ATen/core/Tensor.h>
#include "backend/habana_device/HPUStream.h"

namespace habana_helpers {
at::Tensor cast_tensor_to_integer(const at::Tensor& long_tensor);

at::Tensor cast_tensor_to_long(const at::Tensor& int_tensor);

void copy_scalar_to_host(
    const at::Tensor& src,
    void* dst_ptr,
    uint32_t size,
    c10::hpu::HPUStream hpu_stream);
c10::Scalar _local_scalar_dense_internal(const at::Tensor& self);

at::Tensor downcast_to_int_if_needed(const at::Tensor& in);

at::Tensor hpu_cast_tensor(const at::Tensor& Input, caffe2::TypeMeta type);

} // namespace habana_helpers
