/******************************************************************************
 * Copyright (C) 2023 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/_foreach_add.h"
#include "generated/backend/add.h"
#include "generated/backend/rsub.h"
#include "generated/backend/sub.h"
#define OTHER_INDEX 1
#define ALPHA_INDEX 2

namespace habana {
sizes_vec BinaryOutputShape(const at::Stack& stack) {
  if (stack.at(0).isScalar() && stack.at(OTHER_INDEX).isTensor()) {
    return {stack_tensor(stack, OTHER_INDEX).sizes().vec()};
  }
  const torch::Tensor& self = stack_tensor(stack, 0);
  if (stack.at(OTHER_INDEX).isScalar()) {
    return {self.sizes().vec()};
  }
  const torch::Tensor& other = stack_tensor(stack, OTHER_INDEX);
  return {at::infer_size(self.sizes(), other.sizes())};
}

sizes_vec BinaryOutputShapeInplace(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  return {self.sizes().vec()};
}

enum modes { cadd, csub, crsub };
std::shared_ptr<void> FillBinaryWithAlphaParams(
    const at::Stack& stack,
    size_t& size,
    enum modes mode_t) {
  PARAMS_STUB(ns_BinaryWithAlphaKernel::Params);
  //      params.alpha.i = static_cast<int>(alpha_val);
  params->alpha.f = stack[ALPHA_INDEX].toScalar().toFloat();
  if (mode_t == cadd)
    params->mode = BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_ADD;
  else if (mode_t == csub)
    params->mode = BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_SUB;
  else
    params->mode = BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_RSUB;

  return params;
}
std::shared_ptr<void> FillBinaryRSubParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBinaryWithAlphaParams(stack, size, crsub);
}

std::shared_ptr<void> FillBinarySubParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBinaryWithAlphaParams(stack, size, csub);
}

std::shared_ptr<void> FillBinaryAddParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBinaryWithAlphaParams(stack, size, cadd);
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
    int out_index,
    bool add_casts) {
  std::unique_ptr<synapse_helpers::tensor> constant;
  std::vector<synapse_helpers::tensor> mul, cast;

  if (add_casts) {
    for (auto i = 0u; i < inputs.size(); ++i) {
      if (result_type == dtypes[i]) {
        continue;
      }
      cast.push_back(OpBackend::BuildCast(
          op, graph, inputs[i], sizes[i], dtypes[i], result_type));
      inputs[i] = cast.back().get();
    }
  }

  if (alpha.has_value() and alpha.value().toFloat() != 1.) {
    constant = std::make_unique<synapse_helpers::tensor>(
        OpBackend::BuildConstant(op, graph, *alpha, result_type));
    mul = OpBackend::BuildNode(
        op,
        graph,
        {MULT_GUID + habana_helpers::name_suffix_from_type(result_type),
         {inputs[OTHER_INDEX], constant->get()},
         {{sizes[OTHER_INDEX], result_type}}});
    inputs[OTHER_INDEX] = mul[0].get();
  }

  auto outshape = at::infer_size(sizes[0], sizes[1]);

  return OpBackend::BuildNode(
      op,
      graph,
      {update_guid_dtype(guid, result_type),
       inputs,
       {{outshape, result_type, out_index}}});
}

void ForeachBinary::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& selfs = stack[0].toTensorList();
  if (stack.at(1).isTensorList()) {
    const auto& others = stack[1].toTensorList();
    at::optional<at::Scalar> alpha;
    if (stack.size() > 2) {
      alpha = stack[2].toScalar();
    }
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
          i,
          true);
      syn_out(i) = std::move(out[0]);
    }
  } else {
    for (auto i = 0u; i < selfs.size(); ++i) {
      const auto& other_scalar = stack[OTHER_INDEX].isScalar()
          ? stack[OTHER_INDEX].toScalar()
          : stack[OTHER_INDEX].toListRef()[i].toScalar();
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
          i,
          true);
      syn_out(i) = std::move(out[0]);
    }
  }
}
} // namespace habana
