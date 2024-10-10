/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#pragma once

#include <ATen/core/Tensor.h>

namespace habana_helpers {
at::Tensor cast_tensor_to_integer(const at::Tensor& long_tensor);

at::Tensor cast_tensor_to_long(const at::Tensor& int_tensor);

void copy_scalar_to_host(const at::Tensor& src, void* dst_ptr, uint32_t size);
c10::Scalar _local_scalar_dense_internal(const at::Tensor& self);

at::Tensor downcast_to_int_if_needed(const at::Tensor& in);

at::Tensor hpu_cast_tensor(const at::Tensor& Input, caffe2::TypeMeta type);

} // namespace habana_helpers
