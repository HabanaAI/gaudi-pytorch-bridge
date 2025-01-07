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

#include "hpu_ops/scaled_triangular_softmax.h"

namespace habana {

static auto GetScaledTriangularSoftmaxParams(const float inv_scale_attn) {
  ns_ScaledMaskedSoftmax::Params params{};
  params.invScaleAttn = inv_scale_attn;
  params.groupedBatchSize = 1;
  params.isUseMax = 1;
  params.expMode = USE_LUT;
  return params;
}

ScaledTriangularSoftmax::ScaledTriangularSoftmax(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "scaled_masked_triangular_softmax_fwd",
          scalar_type,
          {0},
          {},
          {},
          false) {}

void ScaledTriangularSoftmax::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "ScaledTriangularSoftmax::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  auto inv_scale_attn = stackGetter.getNextInput<double>();
  auto exp_sum_recpr_opt =
      stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto max_opt = stackGetter.getNextInput<c10::optional<TensorsPair>>();

  TORCH_CHECK(self.pt_t.dim() == 3, "Self tensor must be 3D.");

  TORCH_CHECK(
      (exp_sum_recpr_opt && max_opt) || (!exp_sum_recpr_opt && !max_opt),
      "Inputs max and exp_sum_recpr must be both given or Null.");

  if (exp_sum_recpr_opt) {
    auto exp_sum_recpr_shape = exp_sum_recpr_opt->pt_t.sizes().vec();
    auto max_shape = max_opt->pt_t.sizes().vec();
    TORCH_CHECK(
        exp_sum_recpr_shape == max_shape,
        "exp_sum_recpr and max inputs must have the same shape.");

    auto expected_shape = self.pt_t.sizes().vec();
    expected_shape.back() = 1;
    TORCH_CHECK(
        exp_sum_recpr_shape == expected_shape,
        "exp_sum_recpr and max inputs must have shape [self_shape[0], self_shape[1], 1].");
  }

  ns_ScaledMaskedSoftmax::Params params =
      GetScaledTriangularSoftmaxParams(inv_scale_attn);

  std::vector<synTensor> syn_inputs{self.syn_t};
  if (exp_sum_recpr_opt) {
    syn_inputs.push_back(exp_sum_recpr_opt->syn_t);
    syn_inputs.push_back(max_opt->syn_t);
  }

  auto output = OpBackend::BuildNode(
      this,
      graph,
      {GetGuid(),
       syn_inputs,
       {{self.pt_t.sizes().vec(), ScalarType(), 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
}

ScaledTriangularSoftmaxRetain::ScaledTriangularSoftmaxRetain(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "scaled_masked_triangular_softmax_fwd",
          scalar_type,
          {0, 0, 0},
          {},
          {},
          false) {}

void ScaledTriangularSoftmaxRetain::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(
      this, stack, "ScaledTriangularSoftmaxRetain::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  auto inv_scale_attn = stackGetter.getNextInput<double>();

  TORCH_CHECK(self.pt_t.dim() == 3, "Self tensor must be 3D.");

  auto out_shape = self.pt_t.sizes().vec();
  auto retain_out_shape = out_shape;
  retain_out_shape.back() = 1;

  ns_ScaledMaskedSoftmax::Params params =
      GetScaledTriangularSoftmaxParams(inv_scale_attn);

  auto output = OpBackend::BuildNode(
      this,
      graph,
      {GetGuid(),
       {self.syn_t},
       {{out_shape, ScalarType(), 0},
        {retain_out_shape, at::ScalarType::Float, 1},
        {retain_out_shape, ScalarType(), 2}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
  syn_out(1) = std::move(output[1]);
  syn_out(2) = std::move(output[2]);
}

} // namespace habana

static const auto& ScaledTriangularSoftmaxKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::scaled_triangular_softmax",
            KERNEL_FN_GLOBAL(habana::ScaledTriangularSoftmax))
        .add(
            "hpu::scaled_triangular_softmax_retain",
            KERNEL_FN_GLOBAL(habana::ScaledTriangularSoftmaxRetain));
