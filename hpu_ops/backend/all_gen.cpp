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
  const auto& dtype = at::kFloat;
  const auto& input_shape = self.sizes();
  std::unique_ptr<synapse_helpers::tensor> cast;
  if (dtype != self.scalar_type()) {
    cast = std::make_unique<synapse_helpers::tensor>(OpBackend::BuildCast(
        op, graph, input, input_shape, self.scalar_type(), dtype));
    if (!op->isOutputInfMode()) {
      input = cast->get();
    }
  }
  op->SetScalarType(dtype);

  auto zeros = OpBackend::BuildConstant(op, graph, 0.0f, dtype, input_shape);

  auto not_equal = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("not_equal_fwd", dtype),
       {input, zeros.get()},
       {{input_shape.vec(), dtype}}});

  auto reduce_prod = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      {not_equal[0].get()},
      dim,
      keepdim,
      "reduce_prod_fwd_f32",
      {{final_shape, dtype}});

  return OpBackend::BuildCast(
      op, graph, reduce_prod[0].get(), final_shape, dtype, at::kBool, 0);
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
