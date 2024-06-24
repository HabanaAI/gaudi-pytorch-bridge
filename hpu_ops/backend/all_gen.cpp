/******************************************************************************
 * Copyright (C) 2021-2024 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/all.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {
static auto AllCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synTensor input,
    const at::IntArrayRef dim,
    const bool keepdim,
    const at::IntArrayRef final_shape) {
  std::vector<synapse_helpers::tensor> reduced;
  auto dtype = self.scalar_type();
  auto isIntegralInput = c10::isIntegralType(dtype, true);
  auto reduce_prod_node = [&](const std::vector<synTensor>& input_reduce) {
    return HandleReductionDimAndKeepdim(
        op,
        graph,
        self,
        input_reduce,
        dim,
        keepdim,
        get_guid_with_precision("reduce_prod_fwd", dtype),
        {{final_shape, dtype}});
  };

  if (isIntegralInput) {
    dtype = at::kFloat;
    auto cast = OpBackend::BuildCast(
        op, graph, input, self.sizes(), self.scalar_type(), dtype);
    input = cast.get();
    op->SetScalarType(dtype);
    reduced = reduce_prod_node({input});
  } else {
    auto abs = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("abs_fwd", dtype),
         {input},
         {{self.sizes().vec(), dtype}}});

    auto ceil = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("ceil_fwd", dtype),
         {abs[0].get()},
         {{self.sizes().vec(), dtype}}});
    reduced = reduce_prod_node({ceil[0].get()});
  }

  return OpBackend::BuildCast(
      op, graph, reduced[0].get(), final_shape, dtype, at::kBool, 0);
}

void AllDims::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dims = stack.at(1).toIntList().vec();
  const bool keepdim = stack.at(2).toBool();

  auto out = AllCommon(
      this,
      graph,
      self,
      syn_in(0),
      dims,
      keepdim,
      AllAnyDimMeta(stack)[0].shape);
  syn_out(0) = std::move(out);
}

void AllDim::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  const int64_t dim = stack.at(1).toInt();
  const bool keepdim = stack.at(2).toBool();

  if (self.numel() == 0) {
    auto false_tensor =
        ConstantHelper(graph, true, c10::ScalarType::Bool, {}, 0);
    syn_out(0) = std::move(false_tensor);
  } else {
    auto out = AllCommon(
        this,
        graph,
        self,
        syn_in(0),
        dim,
        keepdim,
        AllAnyDimMeta(stack)[0].shape);
    syn_out(0) = std::move(out);
  }
}

void All::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  if (self.numel() == 0) {
    auto false_tensor =
        ConstantHelper(graph, true, c10::ScalarType::Bool, {}, 0);
    syn_out(0) = std::move(false_tensor);
  } else {
    auto out = AllCommon(
        this, graph, self, syn_in(0), {}, false, AllAnyMeta(stack)[0].shape);
    syn_out(0) = std::move(out);
  }
}
} // namespace habana
