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
class ReductionBackendTemplate : public OpBackend {
  at::optional<uint8_t> m_dim_index;
  at::optional<uint8_t> m_keepdim_index;
  at::optional<uint8_t> m_dtype_index;

  void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override;

 public:
  using OpBackend::OpBackend;

 protected:
  void SetReductionVarsIndices(
      at::optional<uint8_t> dim_index,
      at::optional<uint8_t> keepdim_index,
      at::optional<uint8_t> dtype_index);
};

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

  return {meta};
}

c10::optional<synapse_helpers::tensor> HandleReductionDtype(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synTensor syn_in,
    at::optional<at::ScalarType> dtype);

std::vector<synapse_helpers::tensor> HandleReductionMultiDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor syn_in,
    const std::string& guid,
    c10::IntArrayRef dimsToReduce,
    const int64_t inputRank,
    const bool keepdim,
    std::vector<NodeAttr::NodeOutputAttr> output_attr);

std::vector<synapse_helpers::tensor> HandleReductionDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    std::vector<synTensor> inputs,
    const at::IntArrayRef dims,
    bool keepdim,
    const std::string& guid,
    std::vector<NodeAttr::NodeOutputAttr> output_attr,
    std::function<std::shared_ptr<
        void>(const int64_t, size_t&, int64_t, c10::optional<at::Scalar>)>
        fill_param_fn,
    c10::optional<at::Scalar> ord = c10::nullopt);

std::vector<synapse_helpers::tensor> HandleReductionDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    std::vector<synTensor> inputs,
    const at::IntArrayRef dims,
    bool keepdim,
    const std::string& guid,
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
