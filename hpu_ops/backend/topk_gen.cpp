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
#include "generated/backend/topk.h"
#include "habana_kernels/index_kernels.h"

namespace habana {

sizes_vec TopkOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto k = stack.at(1).isScalar() ? stack.at(1).toInt()
                                  : stack.at(1).toTensor().sizes().vec()[0];
  auto dim_ = stack.at(2).isNone() ? self.dim() : stack.at(2).toInt();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  std::vector<int64_t> shape = self.sizes().vec();
  if (shape.size() > 0) {
    shape[dim] = k;
  }
  return {{shape, shape}};
}

void Topk::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto k = stack.at(1).isScalar() ? stack.at(1).toInt()
                                  : stack.at(1).toTensor().sizes().vec()[0];
  auto dim_ = stack.at(2).isNone() ? self.dim() : stack.at(2).toInt();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  bool largest = stack.at(3).isNone() ? false : stack.at(3).toBool();

  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");

  auto outshape = TopkOutputShape(stack)[0];

  std::vector<synapse_helpers::tensor> result{};

  std::vector<synTensor> syn_inputs{syn_in(0)};
  ns_TopkNodeV2::ParamsV4 params{};
  // It is ok to set params.bsw = k irrespective of static or DS case
  // As per CGUID doc, bsw is ignored if params.kType = K_TENSOR_SHAPE;
  // which is set in DS case.
  params.bsw = k;
  params.axis = self.dim() - dim - 1;
  params.bottomK = !largest;
  params.isVcData = false;

  // The following 3 lines are needed only in dyn shape case.
  // But done hre uncoditionally because, i. is ok to set null pointer
  // inputs. CreateShapeTensorInput has check internally for DS case
  syn_inputs.emplace_back(nullptr);
  syn_inputs.emplace_back(nullptr);
  CreateShapeTensorInput(graph, ScalarType(), outshape, syn_inputs);

  if (graph.is_dynamic_graph()) {
    params.kType = K_TENSOR_SHAPE;
  }

  // Add topk op
  result = BuildOp(
      graph,
      "topk",
      {std::move(syn_inputs)},
      {{outshape, ScalarType(), 0}, {outshape, c10::ScalarType::Int, 1}},
      &params,
      sizeof(params));
  syn_out(0) = std::move(result[0]);
  syn_out(1) = std::move(result[1]);
}
} // namespace habana
