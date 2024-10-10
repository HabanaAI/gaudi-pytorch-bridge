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
#include "generated/backend/_weight_norm_interface.h"
#include "generated/backend/_weight_norm_interface_backward.h"

namespace habana {

namespace sh = synapse_helpers;

sh::tensor NormCommon(
    OpBackend* op,
    sh::graph& graph,
    synTensor input_tensor,
    at::ScalarType dtype,
    const torch::Tensor& self,
    at::IntArrayRef dim,
    const bool keepdim,
    const at::Scalar& ord,
    const std::vector<NodeAttr::NodeOutputAttr>& output_attr,
    const bool is_vec_norm);

OutputMetaDataVector WeightNormMeta(const at::Stack& stack) {
  const torch::Tensor& v_in = stack_tensor(stack, 0);
  const torch::Tensor& g_in = stack_tensor(stack, 1);

  OutputMetaDataVector metaVec(2);
  metaVec[0].shape = v_in.sizes().vec();
  metaVec[1].shape = g_in.sizes().vec();

  metaVec[0].dtype = v_in.scalar_type();
  metaVec[1].dtype = g_in.scalar_type();
  return metaVec;
}

void WeightNormOp::AddNode(sh::graph& graph, const at::Stack& stack) {
  const auto metas = WeightNormMeta(stack);
  auto v_in = stack_tensor(stack, 0);
  auto g_in = stack_tensor(stack, 1);
  auto dim = stack.at(2).toInt();

  const auto& v_in_dtype = metas[0].dtype;
  const auto& g_in_dtype = metas[1].dtype;
  const auto& v_in_shape = metas[0].shape;
  const auto& g_in_shape = metas[1].shape;

  /*
  NOTE:
  We use the CPU implementation that follows the "non-fused" (ie., assumes
  can_use_fused=0) path.
  */
  TORCH_CHECK(
      v_in.device().type() == g_in.device().type(),
      "weight_norm: expected v_in and g_in to be on the same device, but v_in is "
      "on ",
      v_in.device(),
      " and g_in is on ",
      g_in.device());

  c10::DimVector dims_to_norm;
  dims_to_norm.reserve(v_in.ndimension());
  for (int64_t i = 0; i < v_in.ndimension(); ++i) {
    if (i != dim) // skip given dimension
      dims_to_norm.push_back(i);
  }

  at::Scalar ord = 2.0;

  auto normOp = NormCommon(
      this,
      graph,
      g_in_dtype != v_in_dtype
          ? BuildCast(
                this, graph, syn_in(0), v_in.sizes(), v_in_dtype, g_in_dtype)
                .get()
          : syn_in(0),
      g_in_dtype,
      v_in,
      dims_to_norm,
      g_in_shape.size() == v_in_shape.size(),
      ord,
      {{g_in_shape, g_in_dtype, 1}},
      false);

  auto divOp = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", v_in_dtype),
      {syn_in(1), normOp.get()},
      {{g_in_shape, g_in_dtype}});
  auto mulOp = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", v_in_dtype),
      {syn_in(0), divOp.at(0).get()},
      {{v_in_shape, v_in_dtype, 0}});

  syn_out(0) = std::move(mulOp[0]);
  syn_out(1) = std::move(normOp);
}

OutputMetaDataVector WeightNormBwdMeta(const at::Stack& stack) {
  const torch::Tensor& grad_w = stack_tensor(stack, 0);
  const torch::Tensor& saved_v = stack_tensor(stack, 1);
  const torch::Tensor& saved_g = stack_tensor(stack, 2);
  const torch::Tensor& saved_norms = stack_tensor(stack, 3);
  auto dim = stack.at(4).toInt();
  int64_t last_dim = saved_v.dim() - 1;
  int64_t last_size = saved_v.size(last_dim);
  std::vector<int64_t> bcast_size(saved_v.dim(), 1);
  if (dim == 0) {
    bcast_size[0] = saved_v.size(0);
  } else {
    bcast_size[last_dim] = last_size;
  }

  OutputMetaDataVector metaVec(2);
  metaVec[0].shape = at::infer_size(grad_w.sizes(), saved_v.sizes());
  metaVec[1].shape = bcast_size;
  metaVec[0].dtype = saved_v.scalar_type();
  metaVec[1].dtype = saved_g.scalar_type();
  return metaVec;
}
std::shared_ptr<void> FillWeightNormBwdParams(
    const at::Stack& stack,
    size_t& size) {
  auto input = stack.at(0).toTensor();
  auto dim = at::maybe_wrap_dim(stack.at(4).toInt(), input.dim());

  PARAMS_STUB(ns_Reduction::Params);
  params->reductionDimension = dim;
  return params;
}

} // namespace habana
