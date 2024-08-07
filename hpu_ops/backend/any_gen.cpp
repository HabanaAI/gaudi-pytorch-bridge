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

#include "generated/backend/all.h"
#include "generated/backend/any.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {
OutputMetaDataVector AllAnyMeta(const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();

  OutputMetaData meta;
  meta.dtype = at::kBool;
  meta.shape = {};
  return {meta};
}

OutputMetaDataVector AllAnyDimMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const bool keepdim = stack.at(2).toBool();

  OutputMetaData meta;
  if (stack.at(1).isInt()) {
    auto dim = stack.at(1).toInt();
    meta.shape = ReductionOutputShape(self, dim, keepdim)[0];
  } else {
    auto dims = stack.at(1).toIntList().vec();
    meta.shape = ReductionOutputShape(self, dims, keepdim)[0];
  }

  meta.dtype = at::kBool;
  return {meta};
}

SharedMetaDataVector AnySharedMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();

  std::pair<int, at::ScalarType> metaTensor{self.dim(), c10::ScalarType::Float};

  SharedMetaData absMetaData{"abs_fwd"};
  absMetaData.inputs_data = {metaTensor};
  absMetaData.outputs_data = {metaTensor};

  SharedMetaData reduceMetaData{"reduce_sum_multi_dim_fwd"};
  reduceMetaData.inputs_data = {metaTensor};

  int outDim = stack.size() > 1 and stack.at(1).isInt() ? (int)self.dim() : 1;
  reduceMetaData.outputs_data = {{outDim, c10::ScalarType::Float}};

  return {absMetaData, reduceMetaData};
}

static synapse_helpers::tensor AnyCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input,
    const at::Tensor& self,
    const at::IntArrayRef dims,
    const bool keepdim,
    const at::IntArrayRef outshape) {
  // TODO: for integral types, use reduce_sum_fwd_i32 instead
  const auto& dtype = at::kFloat;
  std::unique_ptr<synapse_helpers::tensor> cast;
  if (dtype != self.scalar_type()) {
    cast = std::make_unique<synapse_helpers::tensor>(OpBackend::BuildCast(
        op, graph, input, self.sizes(), self.scalar_type(), dtype));
    if (!op->isOutputInfMode()) {
      input = cast->get();
    }
  }

  op->SetScalarType(dtype);

  auto abs = OpBackend::BuildNode(
      op, graph, {"abs_fwd_f32", {input}, {{self.sizes().vec()}}});

  auto rank = self.dim();
  ns_Reduction::ParamsV2 reductionParams =
      FillReductionParams(rank, dims, keepdim);

  auto reduce_sum = op->BuildNode(
      op,
      graph,
      {get_guid_with_precision("reduce_sum_multi_dim_fwd", dtype),
       {abs[0].get()},
       {{outshape, dtype}},
       &reductionParams,
       sizeof(reductionParams)});

  return OpBackend::BuildCast(
      op, graph, reduce_sum[0].get(), outshape, dtype, at::kBool, 0);
}

void AnyDims::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto dims = stack.at(1).toDimVector();
  bool keepdim = stack.at(2).toBool();

  auto any_out = AnyCommonFunc(
      this,
      graph,
      syn_in(0),
      self,
      dims,
      keepdim,
      AllAnyDimMeta(stack)[0].shape);
  syn_out(0) = std::move(any_out);

  return;
}

void AnyDim::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto dim = stack.at(1).toInt();
  bool keepdim = stack.at(2).toBool();

  auto any_out = AnyCommonFunc(
      this,
      graph,
      syn_in(0),
      self,
      dim,
      keepdim,
      AllAnyDimMeta(stack)[0].shape);
  syn_out(0) = std::move(any_out);
}

void Any::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);

  auto any_out = AnyCommonFunc(
      this, graph, syn_in(0), self, {}, false, AllAnyMeta(stack)[0].shape);
  syn_out(0) = std::move(any_out);
}
} // namespace habana
