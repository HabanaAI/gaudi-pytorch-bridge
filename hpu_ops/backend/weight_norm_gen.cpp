/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "generated/backend/_weight_norm_interface.h"
#include "generated/backend/_weight_norm_interface_backward.h"
#include "hpu_ops/backend/reduction_template.h"

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

static c10::DimVector getDimsToNorm(const int rank, const int dim) {
  c10::DimVector dims_to_norm;
  dims_to_norm.reserve(rank);

  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim) // skip given dimension
      dims_to_norm.push_back(i);
  }

  return dims_to_norm;
}

OutputMetaDataVector WeightNormMeta(const at::Stack& stack) {
  const torch::Tensor& v_in = stack_tensor(stack, 0);
  const torch::Tensor& g_in = stack_tensor(stack, 1);
  auto dim = stack.at(2).toInt();
  const auto keepdim = g_in.sizes().vec().size() == v_in.sizes().vec().size();

  c10::DimVector dims_to_norm = getDimsToNorm(v_in.ndimension(), dim);

  OutputMetaDataVector metaVec(2);
  metaVec[0].shape = v_in.sizes().vec();
  metaVec[1].shape = ReductionOutputShape(v_in, dims_to_norm, keepdim)[0];

  metaVec[0].dtype = v_in.scalar_type();
  metaVec[1].dtype = g_in.scalar_type();
  return metaVec;
}

void moveLastOutputTensorAtFront(OpBackend& op) {
  auto& outputInfMeta = op.GetOutputInfMeta();
  auto output_tensor_idx = outputInfMeta.GetOutputTensor().size() - 1;
  auto output_tensor = outputInfMeta.GetOutputTensor(output_tensor_idx);
  outputInfMeta.RemoveOutput(output_tensor_idx);
  outputInfMeta.PushOutputTensorAtFront(output_tensor);
}
void WeightNormOp::AddNode(sh::graph& graph, const at::Stack& stack) {
  const auto metas = WeightNormMeta(stack);
  auto v_in = stack_tensor(stack, 0);
  auto g_in = stack_tensor(stack, 1);
  auto dim = stack.at(2).toInt();

  const auto& v_dtype = metas[0].dtype;
  const auto& v_shape = metas[0].shape;

  const auto& g_dtype = metas[1].dtype;
  const auto& g_in_shape = g_in.sizes().vec();

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

  c10::DimVector dims_to_norm = getDimsToNorm(v_in.ndimension(), dim);

  at::Scalar ord = 2.0;

  auto normOp = NormCommon(
      this,
      graph,
      g_dtype != v_dtype
          ? BuildCast(this, graph, syn_in(0), v_in.sizes(), v_dtype, g_dtype)
                .get()
          : syn_in(0),
      g_dtype,
      v_in,
      dims_to_norm,
      g_in_shape.size() == v_shape.size(),
      ord,
      {{metas[1].shape, g_dtype, 1}},
      false);

  auto divOp = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", v_dtype),
      {syn_in(1), normOp.get()},
      {{g_in_shape, g_dtype}});
  auto mulOp = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", v_dtype),
      {syn_in(0), divOp.at(0).get()},
      {{v_shape, v_dtype, 0}});

  if (isOutputInfMode()) {
    moveLastOutputTensorAtFront(*this);
  }
  syn_out(0) = std::move(mulOp[0]);
  syn_out(1) = std::move(normOp);
}

OutputMetaDataVector WeightNormBwdMeta(const at::Stack& stack) {
  const torch::Tensor& grad_w = stack_tensor(stack, 0);
  const torch::Tensor& saved_v = stack_tensor(stack, 1);
  const torch::Tensor& saved_g = stack_tensor(stack, 2);
  const torch::Tensor& saved_norms = stack_tensor(stack, 3);
  TORCH_CHECK(grad_w.is_contiguous(), "grad_w must be contiguous");
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");

  auto dim = stack.at(4).toInt();
  int64_t last_dim = saved_v.dim() - 1;

  TORCH_CHECK(
      dim == 0 || dim == last_dim,
      "Expected dim to be the first or last dimension");
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
