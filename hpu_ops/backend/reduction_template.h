/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/lazy_kernels.h"
#include "hpu_ops/common/reduction_template.h"

namespace habana {

inline at::ScalarType get_dtype_from_self(
    const at::Tensor& self,
    const at::optional<at::ScalarType>& dtype,
    bool promote_integers) {
  if (dtype.has_value()) {
    return dtype.value();
  }
  at::ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return at::kLong;
  }
  return src_type;
}

class ReductionBackendTemplate : public OpBackend {
  at::optional<uint8_t> m_dim_index;
  at::optional<uint8_t> m_keepdim_index;
  at::optional<uint8_t> m_dtype_index;

  void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override;

 public:
  ReductionBackendTemplate(
      int device_id,
      const std::string& guid,
      at::ScalarType scalar_type,
      std::vector<int> res_ids,
      std::vector<int> inplace_ids,
      std::vector<int> scalar_ids,
      bool is_outfn);

 protected:
  void SetReductionVarsIndices(
      at::optional<uint8_t> dim_index,
      at::optional<uint8_t> keepdim_index,
      at::optional<uint8_t> dtype_index) {
    m_dim_index = dim_index;
    m_keepdim_index = keepdim_index;
    m_dtype_index = dtype_index;
  }
};

c10::optional<synapse_helpers::tensor> HandleReductionDtype(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synTensor syn_in,
    at::optional<at::ScalarType> dtype);

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
} // namespace habana
