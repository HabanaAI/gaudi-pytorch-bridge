/*******************************************************************************
 * Copyright 2024 Intel Corporation.
 *
 * This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 * This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in
 * the License.
 *******************************************************************************
 */

#pragma once

#include <vector>
#include "habana_eager/ops/as_strided.h"
#include "habana_eager/ops/batch_as_strided.h"

namespace habana::eager {
std::vector<at::Tensor> batch_as_strided(
    at::TensorList inputs,
    c10::ArrayRef<std::vector<int64_t>> sizes,
    c10::ArrayRef<std::vector<int64_t>> strides,
    at::OptionalIntArrayRef storage_offsets);
} // namespace habana::eager
