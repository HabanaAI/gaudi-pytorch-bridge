/**
 * Copyright (c) 2025 Intel Corporation
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

#include <ATen/RedispatchFunctions.h>
#include <torch/torch.h>
#include "backend/synapse_helpers/layout_utils.h"
#include "generated/autograd/autograd_ops.h"

namespace habana {

using namespace synapse_helpers::layouts;

at::Tensor UpsampleBicubic2DCustomVecFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    at::OptionalSymIntArrayRef output_size = at::nullopt,
    bool align_corners = false,
    ::std::optional<at::ArrayRef<double>> scale_factors = at::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;

  ctx->saved_data["input_size"] = input.sizes().vec();
  if (output_size.has_value())
    ctx->saved_data["output_size"] = *output_size;
  else if (scale_factors.has_value())
    ctx->saved_data["scale_factors"] = *scale_factors;
  ctx->saved_data["align_corners"] = align_corners;

  return upsample_bicubic2d_custom_vec_dispatch(
      input, output_size, align_corners, scale_factors);
}

torch::autograd::variable_list UpsampleBicubic2DCustomVecFunction::backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grads) {
  at::AutoDispatchBelowADInplaceOrView guard;

  const auto grad_output = grads[0];
  const auto input_size = ctx->saved_data["input_size"].toIntVector();
  const auto align_corners = ctx->saved_data["align_corners"].toBool();

  at::Tensor grad_input;

  if (ctx->saved_data.find("output_size") != ctx->saved_data.end())
    grad_input = at::redispatch::upsample_bicubic2d_backward(
        c10::DispatchKeySet(c10::DispatchKey::HPU),
        grad_output,
        ctx->saved_data["output_size"].toIntVector(),
        input_size,
        align_corners);
  else if (ctx->saved_data.find("scale_factors") != ctx->saved_data.end()) {
    const auto scale_factors =
        ctx->saved_data["scale_factors"].toDoubleVector();

    grad_input = at::redispatch::upsample_bicubic2d_backward(
        c10::DispatchKeySet(c10::DispatchKey::HPU),
        grad_output,
        {static_cast<int64_t>(input_size[INPUT_H_IDX] * scale_factors[0]),
         static_cast<int64_t>(input_size[INPUT_W_IDX] * scale_factors[1])},
        input_size,
        align_corners,
        scale_factors[0],
        scale_factors[1]);
  }

  return {grad_input, torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

at::Tensor UpsampleTrilinear3DCustomVecFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    at::OptionalSymIntArrayRef output_size = at::nullopt,
    bool align_corners = false,
    ::std::optional<at::ArrayRef<double>> scale_factors = at::nullopt) {
  at::AutoDispatchBelowADInplaceOrView guard;

  ctx->saved_data["input_size"] = input.sizes().vec();
  if (output_size.has_value())
    ctx->saved_data["output_size"] = *output_size;
  else if (scale_factors.has_value())
    ctx->saved_data["scale_factors"] = *scale_factors;
  ctx->saved_data["align_corners"] = align_corners;

  return upsample_trilinear3d_custom_vec_dispatch(
      input, output_size, align_corners, scale_factors);
}

torch::autograd::variable_list UpsampleTrilinear3DCustomVecFunction::backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs) {
  at::AutoDispatchBelowADInplaceOrView guard;

  const auto grad_output = grad_outputs[0];
  const auto input_size = ctx->saved_data["input_size"].toIntVector();
  const auto align_corners = ctx->saved_data["align_corners"].toBool();

  at::Tensor grad_input;

  if (ctx->saved_data.find("output_size") != ctx->saved_data.end())
    grad_input = at::upsample_trilinear3d_backward(
        grad_output,
        ctx->saved_data["output_size"].toIntVector(),
        input_size,
        align_corners);
  else if (ctx->saved_data.find("scale_factors") != ctx->saved_data.end()) {
    const auto scale_factors =
        ctx->saved_data["scale_factors"].toDoubleVector();

    grad_input = at::redispatch::upsample_trilinear3d_backward(
        c10::DispatchKeySet(c10::DispatchKey::HPU),
        grad_output,
        {static_cast<int64_t>(input_size[INPUT_3D_D_IDX] * scale_factors[0]),
         static_cast<int64_t>(input_size[INPUT_3D_H_IDX] * scale_factors[1]),
         static_cast<int64_t>(input_size[INPUT_3D_W_IDX] * scale_factors[2])},
        input_size,
        align_corners,
        scale_factors[0],
        scale_factors[1],
        scale_factors[2]);
  }

  return {grad_input, torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

} // namespace habana
