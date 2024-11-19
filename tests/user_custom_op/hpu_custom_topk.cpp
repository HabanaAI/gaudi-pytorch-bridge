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
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"
#include "synapse_common_types.hpp"

static habana::PartialOutputMetaDataVector output_meta(
    const at::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto k = inputs[1].toInt();
  auto dim = inputs[2].toInt();
  std::vector<int64_t> output_shape = self.sizes().vec();
  if (output_shape.size() > 0) {
    output_shape[dim] = k;
  }
  habana::PartialOutputMetaData meta_output{
      c10::ScalarType::Float, output_shape};
  habana::PartialOutputMetaData meta_indices{
      c10::ScalarType::Long, output_shape};
  return {meta_output, meta_indices};
}

static std::shared_ptr<void> fill_params(
    const at::Stack& inputs,
    size_t& size) {
  HPU_PARAMS_STUB(synBeamParams);
  auto self = inputs[0].toTensor();
  params->bsw = inputs[1].toInt();
  auto dim = inputs[2].toInt();
  params->axis = self.dim() - dim - 1;
  params->bottomK = inputs[3].toBool();
  return params;
}

bool register_custom_topk() {
  habana::custom_op::registerUserCustomOp(
      "custom_op::custom_topk", "topk", output_meta, fill_params);
  return true;
}

std::tuple<at::Tensor, at::Tensor> custom_topk(
    at::Tensor input_a,
    at::Scalar k,
    at::Scalar axis,
    bool bottom) {
  auto op_desc =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          "custom_op::custom_topk");
  std::vector<c10::IValue> inputs{input_a, k, axis, bottom};
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  return {output[0], output[1]};
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def(
      "custom_topk(Tensor self, Scalar k, Scalar axis, bool bottom) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_topk", custom_topk);
}

std::tuple<at::Tensor, at::Tensor> custom_topk_meta(
    at::Tensor input_a,
    at::Scalar k,
    at::Scalar axis,
    bool bottom) {
  auto output_shape = input_a.sizes().vec();
  if (output_shape.size() > 0) {
    output_shape[axis.toInt()] = k.toInt();
  }
  auto output = input_a.new_empty(output_shape, c10::ScalarType::Float);
  auto indices = input_a.new_empty(output_shape, c10::ScalarType::Long);
  return {output, indices};
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("custom_topk", &custom_topk_meta);
}

static const auto& KernelReg = register_custom_topk();
