/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"

namespace habana {
sizes_vec BinaryOutputShape(const at::Stack& stack, bool) {
  if (stack.at(0).isScalar() && stack.at(1).isTensor()) {
    return {stack_tensor(stack, 1).sizes().vec()};
  }
  const torch::Tensor& self = stack_tensor(stack, 0);
  if (stack.at(1).isScalar()) {
    return {self.sizes().vec()};
  }
  const torch::Tensor& other = stack_tensor(stack, 1);
  return {at::infer_size(self.sizes(), other.sizes())};
}

static auto BuildBinary(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::string& guid,
    std::vector<synTensor> inputs,
    sizes_vec sizes,
    const std::vector<at::ScalarType>& dtypes,
    at::ScalarType result_type,
    at::optional<at::Scalar> alpha,
    int out_index) {
  std::unique_ptr<synapse_helpers::tensor> constant;
  std::vector<synapse_helpers::tensor> mul, cast;

  for (auto i = 0u; i < inputs.size(); ++i) {
    if (result_type == dtypes[i]) {
      continue;
    }
    cast.push_back(OpBackend::BuildCast(
        op, graph, inputs[i], sizes[i], dtypes[i], result_type));
    inputs[i] = cast.back().get();
  }

  if (alpha.has_value() and alpha.value().toFloat() != 1.) {
    constant = std::make_unique<synapse_helpers::tensor>(
        OpBackend::BuildConstant(op, graph, *alpha, result_type));
    mul = OpBackend::BuildNode(
        op,
        graph,
        {MULT_GUID + habana_helpers::name_suffix_from_type(result_type),
         {inputs[1], constant->get()},
         {{sizes[1], result_type}}});
    inputs[1] = mul[0].get();
  }

  auto outshape = at::infer_size(sizes[0], sizes[1]);

  return OpBackend::BuildNode(
      op,
      graph,
      {update_guid_dtype(guid, result_type),
       inputs,
       {{outshape, result_type, out_index}}});
}

void BinaryOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const at::Tensor& self = stack_tensor(stack, 0);
  const at::Tensor& other = stack_tensor(stack, 1);
  const at::ScalarType& result_type = at::result_type(self, other);
  auto alpha = stack[2].toScalar();

  auto op = BuildBinary(
      this,
      graph,
      guid_,
      {syn_in(0), syn_in(1)},
      {self.sizes().vec(), other.sizes().vec()},
      {self.scalar_type(), other.scalar_type()},
      result_type,
      alpha,
      0);

  syn_out(0) = std::move(op[0]);
}

void ForeachBinary::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& selfs = stack[0].toTensorList();
  if (stack.at(1).isTensorList()) {
    const auto& others = stack[1].toTensorList();
    auto alpha = stack[2].toScalar();
    for (auto i = 0u; i < selfs.size(); ++i) {
      const auto& self = selfs[i];
      const auto& other = others[i];
      const auto& result_type = at::result_type(self, other);
      auto out = BuildBinary(
          this,
          graph,
          guid_,
          {syn_in(i), syn_in(static_cast<int>(i + selfs.size()))},
          {self.sizes().vec(), other.sizes().vec()},
          {self.scalar_type(), other.scalar_type()},
          result_type,
          alpha,
          i);
      syn_out(i) = std::move(out[0]);
    }
  } else {
    const auto& other_scalar = stack[1].toScalar();
    for (auto i = 0u; i < selfs.size(); ++i) {
      const auto& self = selfs[i];
      const auto& result_type = at::result_type(self, other_scalar);
      auto other = ConstantHelper(graph, other_scalar, result_type);
      auto out = BuildBinary(
          this,
          graph,
          guid_,
          {syn_in(i), other.get()},
          {self.sizes().vec(), {}},
          {self.scalar_type(), result_type},
          result_type,
          c10::nullopt,
          i);
      syn_out(i) = std::move(out[0]);
    }
  }
}
} // namespace habana
