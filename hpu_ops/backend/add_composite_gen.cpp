/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/common/add_composite_gen.h"
#include "generated/backend/addcdiv.h"
#include "generated/backend/addcmul.h"

constexpr int inp_idx = 0; // index of input(self)
constexpr int oth1_idx = 1; // index of other1
constexpr int oth2_idx = 2; // index of other2
constexpr int val_idx = 3; // index of value

namespace habana {

enum modes { mul, div };

std::shared_ptr<void> FillAddCompositeParams(
    const at::Stack& stack,
    enum modes mode_t,
    size_t& size) {
  PARAMS_STUB(ns_BinaryWithAlphaKernel::Params);
  auto value = stack.at(3).toScalar();
  if (stack_tensor(stack, 0).scalar_type() == c10::ScalarType::Int) {
    params->alpha.i = value.to<int32_t>();
  } else {
    params->alpha.f = value.to<float>();
  }
  if (mode_t == mul)
    params->mode = BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CMUL;
  else if (mode_t == div)
    params->mode = BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CDIV;
  return params;
}

std::shared_ptr<void> FillAddcmulParams(const at::Stack& stack, size_t& size) {
  return FillAddCompositeParams(stack, mul, size);
}

std::shared_ptr<void> FillAddcdivParams(const at::Stack& stack, size_t& size) {
  return FillAddCompositeParams(stack, div, size);
}

sizes_vec AddCOpsOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, inp_idx);
  const torch::Tensor& other1 = stack_tensor(stack, oth1_idx);
  const torch::Tensor& other2 = stack_tensor(stack, oth2_idx);
  auto tmp = at::infer_size(self.sizes(), other1.sizes());
  return {at::infer_size(tmp, other2.sizes())};
}

// // void AddCOpBE::AddNode(synapse_helpers::graph& graph, const at::Stack&
// stack) {
// //   const at::Tensor self = stack_tensor(stack, inp_idx);
// //   const at::Tensor other1 = stack_tensor(stack, oth1_idx);
// //   const at::Tensor other2 = stack_tensor(stack, oth2_idx);

// //   std::vector<synapse_helpers::tensor> mul, variable_op;

// sizes_vec AddCOpsOutputShape(const at::Stack& stack) {
//   const torch::Tensor& self = stack_tensor(stack, inp_idx);
//   const torch::Tensor& other1 = stack_tensor(stack, oth1_idx);
//   const torch::Tensor& other2 = stack_tensor(stack, oth2_idx);
//   auto tmp = at::infer_size(self.sizes(), other1.sizes());
//   return {at::infer_size(tmp, other2.sizes())};
// }
} // namespace habana
