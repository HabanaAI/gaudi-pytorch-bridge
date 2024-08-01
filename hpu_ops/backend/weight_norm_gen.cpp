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
    const std::vector<int64_t>& dim,
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

  std::vector<int64_t> dims_to_norm;
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

void WeightNormBwdOp::AddNode(sh::graph& graph, const at::Stack& stack) {
  const torch::Tensor& grad_w = stack_tensor(stack, 0);
  const torch::Tensor& saved_v = stack_tensor(stack, 1);
  const torch::Tensor& saved_g = stack_tensor(stack, 2);
  const torch::Tensor& saved_norms = stack_tensor(stack, 3);
  auto dim = stack.at(4).toInt();

  const auto metas = WeightNormBwdMeta(stack);

  // It is expected that both outputs have the same dtype
  const auto commonOutDtype = metas[0].dtype;

  // In Functions.cpp, the HardshrinkBackward object supplies
  // "grad.contiguous()" as the first argument, so grad_w should be contiguous
  // here. All these checks should succeed:
  TORCH_CHECK(grad_w.is_contiguous(), "grad_w must be contiguous");
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");

  int64_t last_dim = saved_v.dim() - 1;
  int64_t last_size = saved_v.size(last_dim);

  // Like weight_norm_fused_backward, weight_norm_differentiable_backward should
  // only ever be called through a WeightNormFusedBackward object, so we expect
  // that dim == 0 || dim == saved_v.size(-1)
  TORCH_CHECK(
      dim == 0 || dim == last_dim,
      "Expected dim to be the first or last dimension");

  // saved_g and saved_norms are already shaped to broadcast over the correct
  // dimensions

  // ...but saved_norms might be Float when saved_g and saved_v are half.
  // To consider:  saved_norms.to(..., True );

  std::vector<sh::tensor> norms_cast;
  synTensor saved_norms_syn_tensor = syn_in(3);
  if (saved_norms.scalar_type() != commonOutDtype) {
    norms_cast.emplace_back(BuildCast(
        this,
        graph,
        saved_norms_syn_tensor,
        saved_norms.sizes(),
        saved_norms.scalar_type(),
        commonOutDtype));
    saved_norms_syn_tensor = norms_cast[0].get();
  }
  std::vector<sh::tensor> per_dim_sums;
  std::vector<sh::tensor> divOp21;
  std::vector<sh::tensor> mulOp22;
  std::vector<sh::tensor> divOp23;
  std::vector<sh::tensor> mulOp24;
  std::vector<sh::tensor> subOp25;
  std::vector<sh::tensor> grad_v;
  std::vector<sh::tensor> grad_g;
  std::vector<int64_t> bcast_size = metas[1].shape;

  auto outsize_mulOp11 = at::infer_size(grad_w.sizes(), saved_v.sizes());
  auto mulOp11 = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", commonOutDtype),
      {syn_in(0), syn_in(1)},
      {{outsize_mulOp11, commonOutDtype}});

  std::vector<int64_t> reshape_outshape;
  // Analytic backward path using differentiable primitive ops
  if (dim == 0) {
    reshape_outshape.push_back(saved_v.size(0));
    if (grad_w.numel() > saved_v.numel()) {
      reshape_outshape.push_back(grad_w.numel() / saved_v.size(0));
    } else {
      reshape_outshape.push_back(saved_v.numel() / saved_v.size(0));
    }

  } else {
    if (grad_w.numel() > saved_v.numel()) {
      reshape_outshape.push_back(grad_w.numel() / last_size);
    } else {
      reshape_outshape.push_back(saved_v.numel() / last_size);
    }
    reshape_outshape.push_back(last_size);
  }

  auto reshapeOp12 =
      ReshapeHelper(graph, mulOp11[0].get(), reshape_outshape, commonOutDtype);

  ns_Reduction::Params reduce_params{};
  int axis = dim == 0 ? 1 : 0;

  int reductionDimension = reshape_outshape.size() - axis - 1;
  reduce_params.reductionDimension = reductionDimension;

  per_dim_sums = BuildOp(
      graph,
      get_guid_with_precision("reduce_sum_fwd", commonOutDtype),
      {reshapeOp12.get()},
      {{bcast_size, commonOutDtype}},
      &reduce_params,
      sizeof(reduce_params));

  auto outsize_divOp21 = at::infer_size(saved_g.sizes(), saved_norms.sizes());
  divOp21 = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", commonOutDtype),
      {syn_in(2), saved_norms_syn_tensor},
      {{outsize_divOp21, commonOutDtype}});

  mulOp22 = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", commonOutDtype),
      {saved_norms_syn_tensor, saved_norms_syn_tensor},
      {{saved_norms.sizes(), commonOutDtype}});
  divOp23 = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", commonOutDtype),
      {per_dim_sums[0].get(), mulOp22[0].get()},
      {{bcast_size, commonOutDtype}});

  auto outsize_mulOp24 = at::infer_size(saved_v.sizes(), bcast_size);
  mulOp24 = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", commonOutDtype),
      {syn_in(1), divOp23[0].get()},
      {{saved_v.sizes(), commonOutDtype}});

  auto outsize_subOp25 = at::infer_size(grad_w.sizes(), saved_v.sizes());
  subOp25 = BuildOp(
      graph,
      get_guid_with_precision("sub_fwd", commonOutDtype),
      {syn_in(0), mulOp24[0].get()},
      {{outsize_subOp25, commonOutDtype}});

  auto outsize_grad_v = at::infer_size(saved_g.sizes(), outsize_subOp25);
  grad_v = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", commonOutDtype),
      {divOp21[0].get(), subOp25[0].get()},
      {{outsize_grad_v, commonOutDtype, 0}});
  grad_g = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", commonOutDtype),
      {per_dim_sums[0].get(), saved_norms_syn_tensor},
      {{bcast_size, commonOutDtype, 1}});

  syn_out(0) = std::move(grad_v.at(0));
  syn_out(1) = std::move(grad_g.at(0));
}

} // namespace habana
