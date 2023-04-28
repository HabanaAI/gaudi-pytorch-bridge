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

std::shared_ptr<void> FillBinaryRSubParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_BinaryWithAlphaKernel::Params);

  c10::ScalarType self_type = stack_tensor(stack, 0).scalar_type();
  auto other = stack.at(OTHER_INDEX);
  // self_type and other_type to check whether to use alpha as float or int. For
  // integral inputs, alpha shouldn't be float
  c10::ScalarType other_type;
  if (other.isScalar()) {
    other_type = other.toScalar().isFloatingPoint() ? c10::ScalarType::Float
                                                    : c10::ScalarType::Int;
  } else {
    other_type = other.toTensor().scalar_type();
  }

  auto alpha = stack[ALPHA_INDEX].toScalar();
  if ((c10::isIntegralType(self_type, /*includeBool*/ true) &&
       c10::isIntegralType(other_type, /*includeBool*/ true))) {
    HABANA_ASSERT(
        !alpha.isFloatingPoint(),
        "For integral input tensors, argument alpha must not be a floating",
        "point number.");
  }

  if (alpha.isFloatingPoint()) {
    params->alpha.f = stack[ALPHA_INDEX].toScalar().toFloat();
  } else {
    params->alpha.i = stack[ALPHA_INDEX].toScalar().toInt();
  }

  params->mode = BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_RSUB;
  return params;
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

void BinaryWithAlpha::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const at::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> other_size = {};
  at::ScalarType result_type;
  at::ScalarType other_type;
  at::ScalarType self_type = self.scalar_type();

  if (stack.at(1).isTensor()) {
    const at::Tensor& other = stack_tensor(stack, 1);
    other_size = other.sizes().vec();
    result_type = at::result_type(self, other);
    other_type = other.scalar_type();

  } else {
    const auto& other_scalar = stack[1].toScalar();
    result_type = at::result_type(self, other_scalar);
    // other_size remains empty in scalar case
    other_type = result_type;
  }
  auto alpha = stack[2].toScalar();
  if (IsInplace()) {
    TORCH_CHECK(
        result_type == self_type ||
            (c10::isFloatingType(result_type) &&
             c10::isFloatingType(self_type)),
        "result type ",
        result_type,
        " can't be cast to the desired output type ",
        self_type)
  }
  auto op = BuildBinary(
      this,
      graph,
      guid_,
      {syn_in(0), syn_in(1)},
      {self.sizes().vec(), other_size},
      {self_type, other_type},
      result_type,
      alpha,
      0,
      !IsTypePromotion());

  syn_out(0) = std::move(op[0]);
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
      const auto& other_scalar = stack[1].isScalar()
          ? stack[1].toScalar()
          : stack[1].toListRef()[i].toScalar();
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
