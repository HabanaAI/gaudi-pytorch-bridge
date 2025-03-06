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

#include "generated/backend/scaled_triangular_softmax.h"
#include "generated/backend/scaled_triangular_softmax_retain.h"

namespace habana {

std::shared_ptr<void> FillScaledTriangularSoftmaxParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_ScaledMaskedSoftmax::Params);
  params->invScaleAttn = stack.at(1).toScalar().toDouble();
  params->groupedBatchSize = 1;
  params->isUseMax = 1;
  params->expMode = USE_LUT;
  return params;
}

void ScaledTriangularSoftmax::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "ScaledTriangularSoftmax::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  stackGetter.getNextInput<double>(); // inv_scale_attn
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

  size_t size = 0;
  auto params = FillScaledTriangularSoftmaxParams(stack, size);

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
       params.get(),
       size});

  syn_out(0) = std::move(output[0]);
}

OutputMetaDataVector ScaledTriangularSoftmaxRetainMeta(const at::Stack& stack) {
  const auto self = stack[0].toTensor();
  TORCH_CHECK(self.dim() == 3, "Self tensor must be 3D.");
  auto out_shape = self.sizes().vec();
  auto retain_output_shape = out_shape;
  retain_output_shape.back() = 1;

  OutputMetaDataVector meta(3);
  meta[0].shape = out_shape;
  meta[0].dtype = self.scalar_type();
  meta[1].shape = retain_output_shape;
  meta[1].dtype = at::ScalarType::Float;
  meta[2].shape = retain_output_shape;
  meta[2].dtype = self.scalar_type();
  return {meta};
}

} // namespace habana
