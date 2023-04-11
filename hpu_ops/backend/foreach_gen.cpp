/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/_foreach_acos.h"
#include "generated/backend/_foreach_add.h"
#include "generated/backend/_foreach_exp.h"
#include "generated/backend/_foreach_zero.h"

namespace habana {

OutputMetaDataVector ForeachMeta(const at::Stack& stack) {
  auto tensors = stack[0].toTensorList();
  OutputMetaDataVector meta;
  meta.resize(tensors.size());

  for (int i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    meta[i].dtype = tensor.scalar_type();
    meta[i].shape = tensor.sizes().vec();
  }

  return meta;
}

OutputMetaDataVector ForeachBinaryMeta(const at::Stack& stack) {
  auto list1 = stack[0].toTensorList();
  OutputMetaDataVector meta;
  meta.resize(list1.size());

  // Second arg could be tensorlist, scalarlist or scalar
  if (stack.at(1).isList()) {
    const auto& list2 = stack.at(1).toList();
    TORCH_CHECK(
        list1.size() == list2.size(),
        "List1 size: ",
        list1.size(),
        ", != List2 size: ",
        list2.size());
    for (auto i = 0u; i < list1.size(); ++i) {
      auto& m = meta[i];
      const at::Tensor& t1 = list1[i];
      if (list2[i].isTensor()) {
        at::Tensor t2 = list2[i].toTensor();
        m.dtype = at::result_type(t1, t2);
        m.shape = at::infer_size(t1.sizes(), t2.sizes());
      } else {
        m.dtype = at::result_type(t1, list2[i].toScalar());
        m.shape = t1.sizes().vec();
      }
    }
  } else {
    const auto& scalar = stack.at(1).toScalar();
    for (auto i = 0u; i < list1.size(); ++i) {
      auto& m = meta[i];
      const at::Tensor& t1 = list1[i];
      m.dtype = at::result_type(t1, scalar);
      m.shape = t1.sizes().vec();
    }
  }

  return meta;
}

void Foreach::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out = BuildOp(
        graph, guid_, {syn_in(i)}, {{tensor.sizes(), tensor.scalar_type(), i}});
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

} // namespace habana
