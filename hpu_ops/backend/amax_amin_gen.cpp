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

#include "ATen/core/ivalue.h"
#include "generated/backend/amax.h"
#include "generated/backend/amin.h"
#include "generated/backend/aminmax.h"
#include "hpu_ops/backend/reduction_template.h"

namespace sh = synapse_helpers;

namespace habana {
static at::DimVector toDimVector(c10::IValue ival) {
  return ival.isNone()
      ? at::DimVector{}
      : ival.isInt() ? at::DimVector{ival.toInt()} : ival.toDimVector();
}

static OutputMetaDataVector AminmaxMetaCommon(
    const at::Stack& stack,
    int count) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  at::DimVector dim_vec = toDimVector(stack.at(1));
  const bool keepdim = stack.at(2).toBool();

  auto shapes = ReductionOutputShape(self, dim_vec, keepdim);

  OutputMetaData meta;
  meta.shape = shapes[0];
  meta.dtype = self.scalar_type();
  return OutputMetaDataVector(count, meta);
}

OutputMetaDataVector AminmaxMeta(const at::Stack& stack) {
  return AminmaxMetaCommon(stack, 2);
}

OutputMetaDataVector AminAmaxMeta(const at::Stack& stack) {
  return AminmaxMetaCommon(stack, 1);
}

std::shared_ptr<void> FillAminAmaxParams(const at::Stack& stack, size_t& size) {
  auto input = stack.at(0).toTensor();
  auto rank = input.dim();
  at::DimVector dim_vec = toDimVector(stack.at(1));
  auto keepDim = stack.at(2).toBool();

  PARAMS_STUB(ns_Reduction::ParamsV2);
  *params = FillReductionParams(rank, dim_vec, keepDim);

  return params;
}

void Aminmax::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  size_t paramsSize = 0;
  auto params = FillParams(stack, paramsSize);
  const auto meta = OutputMeta(stack)[0];

  c10::optional<sh::tensor> castedInput{};
  // Convert bool tensor to 0x00 and 0x01
  if (self.scalar_type() == c10::ScalarType::Bool) {
    castedInput = BuildBoolCast(
        this, graph, syn_in(0), self.sizes(), c10::ScalarType::Bool);
  }

  std::array<std::string, 2> guids = {
      get_guid_with_precision("reduce_min_multi_dim_fwd", meta.dtype),
      get_guid_with_precision("reduce_max_multi_dim_fwd", meta.dtype),
  };

  for (size_t i = 0; i < guids.size(); ++i) {
    // Leverage autocast feature from CGUID to support integer inputs
    if (c10::isIntegralType(meta.dtype, true)) {
      update_guid_dtype(guids[i], c10::ScalarType::Int);
    }

    auto input =
        castedInput.has_value() ? castedInput.value().get() : syn_in(0);
    auto op = BuildOp(
        graph,
        guids[i],
        {input},
        {{meta.shape, meta.dtype, i}},
        params.get(),
        paramsSize);
    syn_out(i) = std::move(op[0]);
  }
}

void AminAmax::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  size_t paramsSize = 0;
  auto params = FillParams(stack, paramsSize);
  const auto meta = OutputMeta(stack)[0];

  // Leverage autocast feature from CGUID to support integer inputs
  if (c10::isIntegralType(meta.dtype, true)) {
    update_guid_dtype(guid_, c10::ScalarType::Int);
  }

  c10::optional<sh::tensor> castedInput = c10::nullopt;
  // Convert bool tensor to 0x00 and 0x01
  if (self.scalar_type() == c10::ScalarType::Bool) {
    castedInput = BuildBoolCast(
        this, graph, syn_in(0), self.sizes(), c10::ScalarType::Bool);
  }

  auto input = castedInput.has_value() ? castedInput.value().get() : syn_in(0);
  auto op = BuildOp(
      graph,
      GetGuid(),
      {input},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      paramsSize);
  syn_out(0) = std::move(op[0]);
}
} // namespace habana
