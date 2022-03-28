/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "topk_util.h"

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
    int variant) {
  synBeamParams Topk_params{};
  Topk_params.bsw = kvalue;
  Topk_params.axis = reduction_axis;
  Topk_params.bottomK = descending_order;
  if (variant == 1)
    Topk_params.axis = get_dim_in_tpc_order(reduction_axis, ndimension);

  return OpBackend::BuildNode(
      op,
      graph,
      {"topk",
       std::move(input),
       {{topk_outshape, op->ScalarType()}, {topk_outshape, op->ScalarType()}},
       &Topk_params,
       sizeof(Topk_params)});
}

} // namespace habana
