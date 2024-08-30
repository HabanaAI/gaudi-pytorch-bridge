/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/_foreach_pow.h"
#include "generated/backend/pow.h"
#include "hpu_ops/backend/foreach.h"

namespace habana {

static synapse_helpers::tensor PowScalar(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const std::vector<synTensor>&& inputs,
    const at::Scalar& other,
    at::ScalarType scalar_type,
    const std::vector<int64_t>& outshape,
    const int out_index) {
  const float exponent = other.toFloat();
  std::optional<synapse_helpers::tensor> temp;
  NodeAttr node_attr;
  if (exponent == 1.) {
    // fast path for identity
    node_attr = {"identity", {inputs[0]}, {{outshape, scalar_type, out_index}}};
  } else if (exponent == 2.) {
    // fast path for square, mult_fwd_u8/s8_trunc will be used to handle
    // overflow
    node_attr = {
        get_guid_with_precision("mult_fwd", scalar_type),
        {inputs[0], inputs[0]},
        {{outshape, scalar_type, out_index}}};
  } else if (exponent == 3.) {
    const std::string mult_node =
        get_guid_with_precision("mult_fwd", scalar_type);
    temp = std::move(OpBackend::BuildNode(
        op,
        graph,
        {mult_node, {inputs[0], inputs[0]}, {{outshape, scalar_type}}})[0]);
    node_attr = {
        mult_node,
        {temp.value().get(), inputs[0]},
        {{outshape, scalar_type, out_index}}};
  } else {
    std::string guid = get_guid_with_precision("pow_fwd", scalar_type);
    node_attr = {
        guid, {inputs[0], inputs[1]}, {{outshape, scalar_type, out_index}}};
  }
  return std::move(OpBackend::BuildNode(op, graph, std::move(node_attr))[0]);
}

void PowOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);

  syn_out(0) = PowScalar(
      this,
      graph,
      {syn_in(0), syn_in(1)},
      stack[1].toScalar(),
      ScalarType(),
      self.sizes().vec(),
      0);
}

static synapse_helpers::tensor createForeachPowNode(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const std::vector<synTensor>& syn_inputs,
    const std::vector<at::IValue>& pt_inputs,
    int out_index) {
  if (pt_inputs[0].isTensor() && pt_inputs[1].isTensor()) {
    const at::Tensor& self = pt_inputs[0].toTensor();
    const at::Tensor& other = pt_inputs[1].toTensor();

    auto result_type = at::result_type(self, other);
    if (isIntegralType(result_type, true)) {
      result_type = torch::kFloat32;
    }

    auto outshape = at::infer_size(self.sizes(), other.sizes());
    return std::move(OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("pow_fwd", result_type),
         syn_inputs,
         {{outshape, result_type, out_index}}})[0]);
  } else if (pt_inputs[0].isTensor() && pt_inputs[1].isScalar()) {
    const at::Tensor& self = pt_inputs[0].toTensor();
    const at::Scalar& other = pt_inputs[1].toScalar();

    auto result_type = at::result_type(self, other);
    if (isIntegralType(result_type, true)) {
      result_type = torch::kFloat32;
    }

    auto syn_other = OpBackend::BuildConstant(op, graph, other, result_type);
    return PowScalar(
        op,
        graph,
        {syn_inputs[0], syn_other.get()},
        other,
        result_type,
        self.sizes().vec(),
        out_index);
  } else {
    const at::Scalar& self = pt_inputs[0].toScalar();
    const at::Tensor& other = pt_inputs[1].toTensor();

    auto result_type = at::result_type(self, other);
    if (isIntegralType(result_type, true)) {
      result_type = torch::kFloat32;
    }

    auto syn_self = OpBackend::BuildConstant(op, graph, self, result_type);
    return std::move(OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("pow_fwd", result_type),
         {syn_self.get(), syn_inputs[0]},
         {{other.sizes().vec(), result_type, out_index}}})[0]);
  }
}

static SharedMetaDataVector ForeachPowOneIterationSharedMeta(
    const at::Stack& stack) {
  const auto& self = stack.at(0);
  const auto& other = stack.at(1);
  if (self.isTensor() && other.isTensor()) {
    const auto& selfTensor = self.toTensor();
    const auto& otherTensor = other.toTensor();
    auto selfRank = selfTensor.dim();
    auto otherRank = otherTensor.dim();
    auto outputRank = std::max(selfRank, otherRank);
    auto dtype = at::result_type(selfTensor, otherTensor);
    if (isIntegralType(dtype, true))
      dtype = torch::kFloat32;

    SharedMetaData powSharedMeta{"pow_fwd"};
    powSharedMeta.inputs_data = {{selfRank, dtype}, {otherRank, dtype}};
    powSharedMeta.outputs_data = {{outputRank, dtype}};
    return {powSharedMeta};
  } else if (self.isTensor() && other.isScalar()) {
    const auto& selfTensor = self.toTensor();
    auto rank = selfTensor.dim();
    const auto& otherScalar = other.toScalar();
    auto dtype = at::result_type(selfTensor, otherScalar);
    if (isIntegralType(dtype, true))
      dtype = torch::kFloat32;

    const float exponent = otherScalar.toFloat();
    if (exponent == 1.) {
      SharedMetaData identitySharedMeta{"identity"};
      identitySharedMeta.inputs_data.emplace_back(rank, dtype);
      identitySharedMeta.outputs_data = identitySharedMeta.inputs_data;
      return {identitySharedMeta};
    } else if (exponent == 2. || exponent == 3.) {
      SharedMetaData multSharedMeta{"mult_fwd"};
      multSharedMeta.inputs_data = {{rank, dtype}, {rank, dtype}};
      multSharedMeta.outputs_data.emplace_back(rank, dtype);
      return {multSharedMeta};
    } else {
      SharedMetaData powSharedMeta{"pow_fwd"};
      powSharedMeta.inputs_data = {{rank, dtype}, {1, dtype}};
      powSharedMeta.outputs_data = {{rank, dtype}};
      return {powSharedMeta};
    }
  } else {
    const auto& selfScalar = self.toScalar();
    const auto& otherTensor = other.toTensor();
    auto rank = otherTensor.dim();
    auto dtype = at::result_type(selfScalar, otherTensor);
    if (isIntegralType(dtype, true))
      dtype = torch::kFloat32;

    SharedMetaData powSharedMeta{"pow_fwd"};
    powSharedMeta.inputs_data = {{1, dtype}, {rank, dtype}};
    powSharedMeta.outputs_data = {{rank, dtype}};
    return {powSharedMeta};
  }
}

SharedMetaDataVector PowForeachBinarySharedMeta(const at::Stack& stack) {
  SharedMetaCreateFunction sharedMetaCreator = [](const at::Stack& stack) {
    return ForeachPowOneIterationSharedMeta(stack);
  };

  return CommonForeachBinarySharedMeta(stack, sharedMetaCreator);
}

void PowForeachBinary::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  NodeCreateFunction node_creator = [](OpBackend* op,
                                       synapse_helpers::graph& graph,
                                       std::string&,
                                       const std::vector<synTensor>& syn_inputs,
                                       const std::vector<at::IValue>& pt_inputs,
                                       int out_index) {
    return createForeachPowNode(op, graph, syn_inputs, pt_inputs, out_index);
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

} // namespace habana
