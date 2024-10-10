//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
// All Rights Reserved.
//
// Unauthorized copying of this file or any element(s) within it, via any medium
// is strictly prohibited.
// This file contains Habana Labs, Ltd. proprietary and confidential information
// and is subject to the confidentiality and license agreements under which it
// was provided.
//
//===----------------------------------------------------------------------===//

#include "generated/backend/kthvalue.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

std::vector<int64_t> KthvalueOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  int axis = stack.at(2).toInt();
  bool keep_dims = stack.at(3).toBool();

  return ReductionOutputShape(self, axis, keep_dims)[0];
}

std::shared_ptr<void> FillKthvalueParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Kthvalue::Params);
  params->k_value = stack.at(1).toInt();
  params->axis =
      get_dim_in_tpc_order(stack.at(2).toInt(), stack_tensor(stack, 0).dim());
  params->keep_dims = stack.at(3).toBool();

  return params;
}

OutputMetaDataVector KthvalueMeta(const at::Stack& stack) {
  auto input = stack_tensor(stack, 0);
  auto output_shape = KthvalueOutputShape(stack);

  OutputMetaData values_meta, indices_meta;
  values_meta.dtype = input.scalar_type();
  values_meta.shape = output_shape;
  indices_meta.dtype = c10::ScalarType::Long;
  indices_meta.shape = output_shape;
  return {values_meta, indices_meta};
}

} // namespace habana
