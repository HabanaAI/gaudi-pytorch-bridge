/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/scatter_reduce.h"
namespace habana {

const unsigned SELF_INDEX = 0;
const unsigned DIM_INDEX = 1;
const unsigned REDUCE_INDEX = 4;
const unsigned INCLUDE_SELF_INDEX = 5;

std::shared_ptr<void> ScatterReduceParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_ScatterReduceKernel::Params);
  const auto dim = stack.at(DIM_INDEX).toInt();
  auto reduce = stack.at(REDUCE_INDEX).to<c10::string_view>();
  auto baseScatterOp = (reduce == "add" || reduce == "multiply");
  auto includeSelf =
      baseScatterOp ? true : stack.at(INCLUDE_SELF_INDEX).toBool();

  ScatterReduceMode_t mode;

  static const std::unordered_map<c10::string_view, ScatterReduceMode_t>
      reduceModes = {
          {"sum", ScatterReduceMode_t::SCATTER_REDUCE_SUM},
          {"add", ScatterReduceMode_t::SCATTER_REDUCE_SUM},
          {"prod", ScatterReduceMode_t::SCATTER_REDUCE_PROD},
          {"multiply", ScatterReduceMode_t::SCATTER_REDUCE_PROD},
          {"mean", ScatterReduceMode_t::SCATTER_REDUCE_MEAN},
          {"amax", ScatterReduceMode_t::SCATTER_REDUCE_AMAX},
          {"amin", ScatterReduceMode_t::SCATTER_REDUCE_AMIN},
      };

  auto it = reduceModes.find(reduce);
  if (it != reduceModes.end())
    mode = it->second;
  else
    TORCH_CHECK(false, "Unsupported reduce: ", reduce);

  params->dim = dim;
  params->include_self = includeSelf;
  params->mode = mode;

  return params;
}

OutputMetaDataVector ScatterReduceMeta(const at::Stack& stack) {
  auto self = stack.at(SELF_INDEX);
  std::vector<int64_t> outputShape;
  at::ScalarType dtype;

  if (self.isTensor()) {
    auto selfTensor = self.toTensor();
    outputShape = selfTensor.sizes().vec();
    dtype = selfTensor.scalar_type();
  } else {
    dtype = self.toScalar().type();
  }

  OutputMetaData meta;
  meta.dtype = dtype;
  meta.shape = outputShape;

  return {meta};
}

} // namespace habana
