/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <ATen/native/ReduceOpsUtils.h>
#include "habana_kernels/lazy_kernels.h"
#include "hpu_op_helper.h"

namespace habana {

static inline at::ScalarType get_dtype_from_self(
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

sizes_vec ReductionOutputShape(
    const at::Tensor& self,
    at::optional<at::IntArrayRef> dims,
    bool keepdim);

template <typename T>
class ReductionFrontendTemplate : public habana_lazy::LazyOp<T> {
  at::optional<uint8_t> m_dtype_index;
  at::optional<uint8_t> m_dim_index;
  at::optional<uint8_t> m_keepdim_index;

 public:
  ReductionFrontendTemplate(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
      : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn, -1) {}

  ReductionFrontendTemplate(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const sizes_vec& out_shapes)
      : habana_lazy::LazyOp<T>(qualstring, inputs, {}, out_shapes, -1) {}

  T get_result_overrideable() override;

  void SetReductionVarsIndices(
      at::optional<uint8_t> dim_index,
      at::optional<uint8_t> keepdim_index,
      at::optional<uint8_t> dtype_index) {
    m_dim_index = dim_index;
    m_keepdim_index = keepdim_index;
    m_dtype_index = dtype_index;
  }
};

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

synapse_helpers::tensor HandleReductionDtype(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synapse_helpers::tensor syn_in,
    at::optional<at::ScalarType> dtype);

std::vector<synapse_helpers::tensor> HandleReductionDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    std::vector<synTensor> inputs,
    const at::IntArrayRef dims,
    bool keepdim,
    const std::string& guid,
    const std::vector<NodeAttr::NodeOutputAttr>& output_attr);

} // namespace habana
