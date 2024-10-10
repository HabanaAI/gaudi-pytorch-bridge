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

#include "hpu_ops/topk_util.h"
#include "generated/backend/topk.h"
namespace habana {

std::vector<synapse_helpers::tensor> TopK_Helper(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    int reduction_axis,
    const at::IntArrayRef topk_outshape,
    int descending_order,
    int ndimension,
    int kvalue,
    int variant,
    c10::optional<at::ScalarType> out_dtype) {
  synBeamParams Topk_params{};
  Topk_params.bsw = kvalue;
  Topk_params.axis = reduction_axis;
  Topk_params.bottomK = descending_order;
  auto indices_dtype =
      common::IsInt64Supported() ? c10::ScalarType::Long : c10::ScalarType::Int;
  if (variant == 1)
    Topk_params.axis = get_dim_in_tpc_order(reduction_axis, ndimension);
  at::ScalarType topk_dtype =
      (out_dtype == c10::nullopt) ? op->ScalarType() : out_dtype.value();
  return OpBackend::BuildNode(
      op,
      graph,
      {"topk",
       std::move(input),
       {{topk_outshape, topk_dtype}, {topk_outshape, indices_dtype}},
       &Topk_params,
       sizeof(Topk_params)});
}

} // namespace habana
