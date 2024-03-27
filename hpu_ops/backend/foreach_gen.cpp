/******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/_foreach_abs.h"
#include "generated/backend/_foreach_add.h"
#include "generated/backend/_foreach_div.h"
#include "generated/backend/_foreach_zero.h"
#include "hpu_ops/backend/foreach.h"

namespace habana {
const unsigned SELF_INDEX = 0;
const unsigned OTHER_INDEX = 1;
const unsigned ALPHA_INDEX = 2;

OutputMetaDataVector ForeachMeta(const at::Stack& stack) {
  auto tensors = stack[0].toTensorList();
  OutputMetaDataVector meta;
  meta.resize(tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    meta[i].dtype = tensor.scalar_type();
    meta[i].shape = tensor.sizes().vec();
  }

  return meta;
}

static OutputMetaData MetaForSingleOutput(
    const at::Tensor& self,
    const at::IValue& other,
    const bool cast_int_to_float) {
  OutputMetaData meta;
  if (other.isTensor()) {
    const auto& other_tensor = other.toTensor();
    meta.dtype = at::result_type(self, other_tensor);
    meta.shape = at::infer_size(self.sizes(), other_tensor.sizes());
  } else {
    meta.dtype = at::result_type(self, other.toScalar());
    meta.shape = self.sizes().vec();
  }
  if (cast_int_to_float && isIntegralType(meta.dtype, true)) {
    meta.dtype = torch::kFloat32;
  }
  return meta;
}

OutputMetaDataVector CommonForeachBinaryMeta(
    const at::Stack& stack,
    const bool cast_int_to_float) {
  auto list1 = stack[0].toTensorList();
  OutputMetaDataVector meta;
  meta.resize(list1.size());

  // Second arg could be tensorlist, scalarlist, tensor or scalar
  if (stack.at(1).isList()) {
    const auto& list2 = stack.at(1).toList();
    TORCH_CHECK(
        list1.size() == list2.size(),
        "List1 size: ",
        list1.size(),
        ", != List2 size: ",
        list2.size());
    for (size_t i = 0; i < list1.size(); ++i) {
      meta[i] = MetaForSingleOutput(list1[i], list2[i], cast_int_to_float);
    }
  } else {
    for (size_t i = 0; i < list1.size(); ++i) {
      meta[i] = MetaForSingleOutput(list1[i], stack.at(1), cast_int_to_float);
    }
  }
  return meta;
}

OutputMetaDataVector DivForeachBinaryMeta(const at::Stack& stack) {
  return CommonForeachBinaryMeta(stack, true);
}

OutputMetaDataVector ForeachBinaryMeta(const at::Stack& stack) {
  return CommonForeachBinaryMeta(stack, false);
}

void Foreach::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  size_t params_size = 0;
  auto params = FillParams(stack, params_size);
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out = BuildOp(
        graph,
        guid_,
        {syn_in(i)},
        {{tensor.sizes(), tensor.scalar_type(), i}},
        params.get(),
        params_size);
    syn_out(i) = std::move(out[0]);
  }
}

void ForeachZero::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out =
        ConstantHelper(graph, 0, tensor.scalar_type(), tensor.sizes(), i);
    syn_out(i) = std::move(out);
  }
}

size_t computeInputsNumber(const at::Stack& stack) {
  const size_t self_size = stack[SELF_INDEX].toTensorList().size();
  const size_t other_size = stack[OTHER_INDEX].isTensorList()
      ? self_size
      : stack[OTHER_INDEX].isTensor() ? 1 : 0;
  return self_size + other_size;
}

std::vector<synapse_helpers::tensor> CommonForeachBinary(
    OpBackend* op,
    std::string& guid,
    const std::vector<synTensor>& inputs,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    NodeCreateFunction node_creator) {
  std::vector<synapse_helpers::tensor> outputs;
  const auto& selfs = stack[SELF_INDEX].toTensorList();
  const auto& others = stack[OTHER_INDEX];
  at::optional<at::Scalar> alpha;

  if (stack.at(1).isTensorList() || stack.at(1).isTensor()) {
    if (stack.size() > 2) {
      alpha = stack[ALPHA_INDEX].toScalar();
    }
    for (size_t i = 0; i < selfs.size(); ++i) {
      const auto& self = selfs[i];
      const auto& other = others.isList() ? others.toList()[i] : others;
      const size_t other_syn_index =
          others.isTensorList() ? i + selfs.size() : selfs.size();

      std::vector<at::IValue> pt_inputs = {self, other};
      if (alpha.has_value()) {
        pt_inputs.push_back(alpha.value());
      }

      outputs.push_back(node_creator(
          op, graph, guid, {inputs[i], inputs[other_syn_index]}, pt_inputs, i));
    }
  } else {
    for (size_t i = 0; i < selfs.size(); ++i) {
      const auto& self = selfs[i];
      const auto& other = others.isList() ? others.toList()[i] : others;

      outputs.push_back(
          node_creator(op, graph, guid, {inputs[i]}, {self, other}, i));
    }
  }
  return outputs;
}

} // namespace habana
