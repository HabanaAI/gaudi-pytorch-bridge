/*******************************************************************************
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
#include "hpu_ops/backend/reduction_template_cguid.h"
#include "backend/helpers/lowering_util.h"
#include "hpu_ops/common/reduction_template.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

static std::shared_ptr<void> FillReductionParams(
    int ndims,
    const at::IntArrayRef dims,
    bool keepdim,
    size_t& size) {
  PARAMS_STUB(ns_Reduction::ParamsV2);
  std::string str_dim = std::string(ndims, '0');
  for (int i = 0; i < dims.size(); ++i) {
    str_dim.replace(dims[i], 1, "1");
  }

  unsigned int binary_int = std::stoi(str_dim, nullptr, 2);
  std::bitset<8> bits(binary_int);
  params->reductionDimensionMask = binary_int;
  params->keepDim = keepdim;
  return params;
}

// Returns the input after cast to the supplied dtype. If dtype is none or if
// dtype is same as input's dtype, returns nullopt.
static c10::optional<synapse_helpers::tensor> HandleReductionDtype(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synTensor syn_in,
    at::optional<at::ScalarType> dtype) {
  auto dtype_val = dtype.value_or(self.scalar_type());

  if (!dtype.has_value() and at::isIntegralType(self.scalar_type(), true)) {
    dtype_val = at::kInt;
  }

  if (habana_helpers::getInternalDtype(dtype_val) ==
      habana_helpers::getInternalDtype(self.scalar_type())) {
    return c10::nullopt;
  }

  op->SetScalarType(dtype_val);

  // Update guid with the dtype to be used
  std::string guid = op->GetGuid();
  op->SetGuid(update_guid_dtype(guid, dtype_val));

  return OpBackend::BuildCast(
      op, graph, syn_in, self.sizes(), self.scalar_type(), dtype_val);
}

static void ProcessDim(std::vector<int64_t>& dims, const int& ndims) {
  // When dim=[], reduce all dimensions based on keepdim value
#if 1
  if (0 == dims.size()) {
    for (int i = 0; i < ndims; ++i) {
      dims.push_back(i);
    }
  }
#endif
  if (dims.size()) {
    LoweringUtil::SortAndRemoveDuplicateDims(dims, ndims);
  }
}

ReductionBackendTemplateCGUID::ReductionBackendTemplateCGUID(
    int device_id,
    const std::string& guid,
    at::ScalarType scalar_type,
    std::vector<int> res_ids,
    std::vector<int> inplace_ids,
    std::vector<int> scalar_ids,
    bool is_outfn)
    : OpBackend(
          device_id,
          guid,
          scalar_type,
          std::move(res_ids),
          std::move(inplace_ids),
          std::move(scalar_ids),
          is_outfn) {}

void ReductionBackendTemplateCGUID::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  synTensor input = syn_in(0);

  // Extract dtype and cast self to the supplied dtype
  auto dtype = get_dtype(stack, m_dtype_index);
  auto cast = HandleReductionDtype(this, graph, self, input, dtype);
  if (cast.has_value()) {
    input = cast.value().get();
  }

  // Extract dims
  auto dims = get_dims(stack, m_dim_index);

  // Extract keepdim
  bool keepdim = get_keepdim(stack, m_keepdim_index);

  auto shape = ComputeOutputShapes(stack).empty()
      ? ReductionOutputShape(self, dims, keepdim)[0]
      : ComputeOutputShapes(stack)[0];

  auto guid = GetGuid();
  std::vector<int64_t> orig_shape{self.sizes().vec()};
  int ndims = orig_shape.size();

  ProcessDim(dims, ndims);

  NodeAttr::NodeOutputAttr reduction_node_output_attr = {
      shape, ScalarType(), 0};

  size_t size = 0;
  auto params = FillReductionParams(ndims, dims, keepdim, size);
  auto result = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       {std::move(input)},
       {reduction_node_output_attr},
       params.get(),
       size});

  syn_out(0) = std::move(result[0]);
}
} // namespace habana
