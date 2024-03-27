/*******************************************************************************
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

#include "backend/helpers/cast_sequence.h"
#include "generated/backend/_foreach_add.h"
#include "generated/backend/_foreach_div.h"
#include "generated/backend/add.h"
#include "generated/backend/rsub.h"
#include "generated/backend/sub.h"
#include "hpu_ops/backend/foreach.h"

namespace habana {
const unsigned SELF_INDEX = 0;
const unsigned OTHER_INDEX = 1;
const unsigned ALPHA_INDEX = 2;

sizes_vec BinaryOutputShape(const at::Stack& stack) {
  if (stack.at(0).isScalar() && stack.at(OTHER_INDEX).isTensor()) {
    return {stack_tensor(stack, OTHER_INDEX).sizes().vec()};
  }
  const torch::Tensor& self = stack_tensor(stack, SELF_INDEX);
  if (stack.at(OTHER_INDEX).isScalar()) {
    return {self.sizes().vec()};
  }
  const torch::Tensor& other = stack_tensor(stack, OTHER_INDEX);
  return {at::infer_size(self.sizes(), other.sizes())};
}

sizes_vec BinaryOutputShapeInplace(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, SELF_INDEX);
  return {self.sizes().vec()};
}

std::shared_ptr<void> FillBinaryWithAlphaParams(
    const at::Stack& stack,
    size_t& size,
    BinaryWithAlphaMode_t mode) {
  PARAMS_STUB(ns_BinaryWithAlphaKernel::Params);
  auto self = stack.at(SELF_INDEX);
  auto other = stack.at(OTHER_INDEX);
  at::ScalarType selfType =
      self.isScalar() ? self.toScalar().type() : self.toTensor().scalar_type();
  at::ScalarType otherType = other.isScalar() ? other.toScalar().type()
                                              : other.toTensor().scalar_type();
  auto alpha = stack.at(ALPHA_INDEX).toScalar();

  if ((c10::isIntegralType(selfType, true) &&
       c10::isIntegralType(otherType, true))) {
    HABANA_ASSERT(
        !alpha.isFloatingPoint(),
        "For integral input tensors, argument alpha must not be a floating",
        "point number.");
    params->alpha.i = alpha.to<int>();
  } else
    params->alpha.f = static_cast<float>(alpha.to<double>());

  params->mode = mode;
  return params;
}

std::shared_ptr<void> FillBinaryRSubParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBinaryWithAlphaParams(
      stack, size, BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_RSUB);
}

std::shared_ptr<void> FillBinarySubParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBinaryWithAlphaParams(
      stack, size, BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_SUB);
}

std::shared_ptr<void> FillBinaryAddParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBinaryWithAlphaParams(
      stack, size, BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_ADD);
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
    auto result_cast_type = habana_helpers::DataTypeToCastType(result_type);
    for (auto i = 0u; i < inputs.size(); ++i) {
      if (result_cast_type == habana_helpers::DataTypeToCastType(dtypes[i])) {
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
        {get_guid_with_precision("mult", result_type),
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

static void update_result_type(
    at::ScalarType& result_type,
    std::string& guid,
    bool cast_int_to_float,
    bool support_int8,
    bool support_int16) {
  bool dtype_changed = false;

  if (cast_int_to_float && isIntegralType(result_type, true)) {
    result_type = torch::kFloat32;
    dtype_changed = true;
  } else {
    if (!support_int8 &&
        (result_type == torch::kInt8 || result_type == torch::kUInt8)) {
      result_type = torch::kInt16;
      dtype_changed = true;
    }
    if (!support_int16 && result_type == torch::kInt16) {
      result_type = torch::kInt32;
      dtype_changed = true;
    }
  }

  if (dtype_changed) {
    guid = update_guid_dtype(guid, result_type);
  }
}

static synapse_helpers::tensor createForeachBinaryNode(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::string& guid_,
    const std::vector<synTensor>& syn_inputs,
    const std::vector<at::IValue>& pt_inputs,
    int out_index,
    bool cast_int_to_float = false,
    bool support_int8 = true,
    bool support_int16 = true) {
  const at::Tensor& self = pt_inputs[0].toTensor();
  sizes_vec sizes = {self.sizes().vec()};
  std::vector<synTensor> inputs = syn_inputs;
  std::vector<at::ScalarType> dtypes = {self.scalar_type()};

  at::optional<at::Scalar> alpha = c10::nullopt;
  at::ScalarType result_type;
  at::optional<synapse_helpers::tensor> scalar = c10::nullopt;

  if (pt_inputs[1].isTensor()) {
    const at::Tensor& other = pt_inputs[1].toTensor();
    if (pt_inputs.size() > 2) {
      alpha = pt_inputs[2].toScalar();
    }
    result_type = at::result_type(self, other);
    update_result_type(
        result_type, guid_, cast_int_to_float, support_int8, support_int16);

    sizes.push_back(other.sizes().vec());
    dtypes.push_back(other.scalar_type());
  } else {
    const at::Scalar& other = pt_inputs[1].toScalar();
    result_type = at::result_type(self, other);
    update_result_type(
        result_type, guid_, cast_int_to_float, support_int8, support_int16);

    scalar = OpBackend::BuildConstant(op, graph, other, result_type);
    inputs.push_back(scalar.value().get());

    sizes.push_back({});
    dtypes.push_back(result_type);
  }

  return std::move(BuildBinary(
      op,
      graph,
      guid_,
      inputs,
      sizes,
      dtypes,
      result_type,
      alpha,
      out_index,
      true)[0]);
}

void ForeachBinary::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  bool cast_int_to_float = guid_.find("div") != std::string::npos;
  bool not_min_or_max = guid_.find("min") == std::string::npos &&
      guid_.find("max") == std::string::npos;
  bool support_int8 = guid_.find("sub") == std::string::npos && not_min_or_max;
  bool support_int16 = not_min_or_max;

  NodeCreateFunction node_creator =
      [cast_int_to_float, support_int8, support_int16](
          OpBackend* op,
          synapse_helpers::graph& graph,
          std::string& guid_,
          const std::vector<synTensor>& syn_inputs,
          const std::vector<at::IValue>& pt_inputs,
          int out_index) {
        return createForeachBinaryNode(
            op,
            graph,
            guid_,
            syn_inputs,
            pt_inputs,
            out_index,
            cast_int_to_float,
            support_int8,
            support_int16);
      };

  const size_t size = computeInputsNumber(stack);
  std::vector<synTensor> inputs(size);
  for (size_t i = 0; i < size; i++) {
    inputs[i] = syn_in(i);
  }
  auto results =
      CommonForeachBinary(this, guid_, inputs, graph, stack, node_creator);
  for (size_t i = 0; i < results.size(); i++) {
    syn_out(i) = std::move(results[i]);
  }
}

void BinaryWithAlpha::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const at::Tensor& self = stack_tensor(stack, SELF_INDEX);
  auto other = stack.at(OTHER_INDEX);
  at::ScalarType result_type;
  size_t size = 0;

  auto params = FillParams(stack, size);
  const auto outputShape = IsInplace() ? BinaryOutputShapeInplace(stack)[0]
                                       : BinaryOutputShape(stack)[0];

  bool isAlphaIntegralType{false};
  if (other.isTensor()) {
    isAlphaIntegralType =
        c10::isIntegralType(other.toTensor().scalar_type(), true) &&
        c10::isIntegralType(self.scalar_type(), true);

    const at::Tensor& other_tensor = stack_tensor(stack, OTHER_INDEX);
    result_type = at::result_type(self, other_tensor);
  } else {
    isAlphaIntegralType = c10::isIntegralType(other.toScalar().type(), true) &&
        c10::isIntegralType(self.scalar_type(), true);

    const auto& other_scalar = stack.at(OTHER_INDEX).toScalar();
    result_type = at::result_type(self, other_scalar);
  }

  const auto& filledParams =
      std::reinterpret_pointer_cast<ns_BinaryWithAlphaKernel::Params>(params);
  const auto& alpha = filledParams->alpha;
  const auto& mode = filledParams->mode;

  // binary_with_alpha_fwd* do not support i8 and u8, so we use
  // binary_with_alpha_fwd_i16.
  switch (result_type) {
    case c10::ScalarType::Char:
    case c10::ScalarType::Byte:
    case c10::ScalarType::Bool:
      result_type = c10::ScalarType::Short;
      break;
    default:
      break;
  }

  std::vector<synTensor> inputs{syn_in(SELF_INDEX), syn_in(OTHER_INDEX)};
  std::string guid{guid_};

  if ((isAlphaIntegralType ? alpha.i : alpha.f) == 1) {
    std::string opName;
    switch (mode) {
      case BINARY_WITH_ALPHA_MODE_ADD:
        opName = "add";
        break;
      case BINARY_WITH_ALPHA_MODE_RSUB:
        // RSUB uses SUB kernel, but with reversed inputs
        inputs = {syn_in(OTHER_INDEX), syn_in(SELF_INDEX)};
        [[fallthrough]];
      case BINARY_WITH_ALPHA_MODE_SUB:
        opName = "sub";
        break;
      default:
        opName = {};
    }
    guid = get_guid_with_precision(opName, result_type);
  }
  if ((isAlphaIntegralType ? alpha.i : alpha.f) != 1 &&
      result_type == c10::ScalarType::Short) {
    std::string opName = "binary_with_alpha_fwd";
    guid = get_guid_with_precision(opName, result_type);
  }
  auto op = BuildOp(
      graph,
      std::move(guid),
      std::move(inputs),
      {{outputShape, result_type, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(op[0]);
}

// The same as the native aten.foreach_add_, but we need to disable
// eager compiler for its usage only in accumulate_grads_ op.
struct CustomForeachAdd : ForeachBinary {
  CustomForeachAdd(int device_id, c10::ScalarType scalar_type)
      : ForeachBinary(device_id, "add_fwd", scalar_type, {}, {0}, {}, false) {}
};

} // namespace habana

static const auto& ForeachKernelRegistry = habana::KernelRegistry().add(
    "hpu::custom_foreach_add_",
    KERNEL_FN_GLOBAL(habana::CustomForeachAdd));
