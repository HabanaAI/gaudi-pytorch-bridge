/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace sh = synapse_helpers;

namespace habana {

namespace fp8 {
auto GetFp8Dtypes(const at::ScalarType& dtype);

auto GetFp8Dtypes(const at::IValue& dtype);

void ValidateScaleShape(
    const c10::IValue& scale,
    const c10::IValue& scale_shape);

void HandleScaleTensor(
    habana::OpBackend* op,
    sh::graph& graph,
    const at::Tensor& scale,
    synTensor syn_scale,
    std::vector<sh::tensor>& maybe_reshaped_scale,
    std::vector<synTensor>& syn_inputs,
    const c10::IValue& scale_shape_ival = c10::IValue{});

void HandleScaleScalar(
    habana::OpBackend* op,
    sh::graph& graph,
    const c10::IValue& scale,
    const int device_id,
    std::vector<sh::tensor>& maybe_const_scale,
    std::vector<synTensor>& syn_inputs,
    const c10::IValue& scale_shape_ival = c10::IValue{});
} // namespace fp8

} // namespace habana
