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
#pragma once
#include <ATen/native/ReduceOpsUtils.h>
#include "hpu_ops/common/reduction_template.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

template <int dim_index, int keepdim_index, int dtype_index>
OutputMetaDataVector ReductionMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto convert_index = [](int index) {
    return index < 0 ? c10::nullopt : c10::make_optional<uint8_t>(index);
  };

  auto dims = get_dims(stack, convert_index(dim_index));
  bool keepdim = get_keepdim(stack, convert_index(keepdim_index));

  OutputMetaData meta{};
  meta.shape = ReductionOutputShape(self, dims, keepdim)[0];

  auto dtype = get_dtype(stack, convert_index(dtype_index));
  if (dtype.has_value()) {
    meta.dtype = dtype.value();
  } else if (at::isIntegralType(self.scalar_type(), true)) {
    meta.dtype = at::kLong;
  } else {
    meta.dtype = self.scalar_type();
  }

  // When the output tensor is provided, it indicates this is sum.out version,
  // and output tensor type should be used.
  if (stack.back().isTensor())
    meta.dtype = stack.back().toTensor().scalar_type();

  return {meta};
}

inline bool reduction_support_f32(const std::string& guid) {
  return guid.find("reduce_prod_multi_dim_fwd") != std::string::npos or
      guid.find("reduce_mean_multi_dim_fwd") != std::string::npos;
}

inline bool reduction_support_i32(const std::string& guid) {
  return guid.find("reduce_sum_multi_dim") != std::string::npos;
}

c10::optional<synapse_helpers::tensor> HandleReductionDtype(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synTensor syn_in,
    at::optional<at::ScalarType> dtype);

std::vector<synapse_helpers::tensor> HandleReduction(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor syn_in,
    const std::string& guid,
    c10::IntArrayRef dimsToReduce,
    const int64_t inputRank,
    const bool keepdim,
    std::vector<NodeAttr::NodeOutputAttr> output_attr);

std::vector<int64_t> CalculateReductionMultiDimAndKeepdimOutputSize(
    const std::vector<int64_t>& inputSize,
    const std::vector<int64_t>& dimsToReduce,
    bool keepDim);

ns_Reduction::ParamsV2 FillReductionParams(
    int64_t ndims,
    c10::IntArrayRef dims,
    bool keepdim);

} // namespace habana
