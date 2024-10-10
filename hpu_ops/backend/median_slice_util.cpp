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
#include "hpu_ops/median_slice_util.h"

namespace habana {

std::vector<synapse_helpers::tensor> Median_Slice_Helper(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> inputs,
    const at::IntArrayRef outshape,
    const at::ScalarType dtype,
    int64_t nelements,
    int64_t ndimension,
    int64_t reduction_axis,
    int64_t median_variant,
    bool final_node,
    c10::optional<int> node_index) {
  if (!final_node)
    node_index = c10::nullopt;

  synDynamicSliceDmaH2dTensor sliceDs{};
  bool useDsVariant = graph.is_dynamic_graph() || graph.is_eager_mode();

  // TODO: SW-166787 enable it
  useDsVariant = false;

  if (useDsVariant) {
    sliceDs.dims = outshape.size();
    for (unsigned d = 0; d < sliceDs.dims; ++d) {
      sliceDs.steps[d] = 1;
    }
  }

  synSliceParamsV2 slice_params{};
  if (median_variant == 0) {
    slice_params.axes[0] = reduction_axis;
    slice_params.starts[0] = nelements / 2;
    slice_params.ends[0] = nelements / 2;
    slice_params.steps[0] = 1;
    if (useDsVariant) {
      sliceDs.starts[reduction_axis] = slice_params.starts[0];
    }
  } else {
    for (int64_t idx = 0; idx < ndimension; ++idx) {
      slice_params.axes[idx] = idx;
      slice_params.steps[idx] = 1;
      if (idx == (get_dim_in_tpc_order(reduction_axis, ndimension))) {
        slice_params.starts[idx] = nelements / 2;
        slice_params.ends[idx] = nelements / 2;
        if (useDsVariant) {
          sliceDs.starts[idx] = slice_params.starts[idx];
        }
      } else {
        slice_params.starts[idx] = 0;
        slice_params.ends[idx] =
            outshape[get_dim_in_tpc_order(idx, ndimension)];
      }
    }
  }

  if (useDsVariant) {
    op->CreateShapeTensorInput(
        graph, dtype, outshape, inputs, SHAPE_TENSOR, graph.is_eager_mode());

    op->CreateH2dTensorInput(
        graph,
        c10::ScalarType::Int,
        &sliceDs,
        sizeof(sliceDs),
        inputs,
        HOST_TO_DEVICE_TENSOR,
        graph.is_eager_mode());
  }

  return OpBackend::BuildNode(
      op,
      graph,
      {"slice",
       std::move(inputs),
       {{outshape, dtype, node_index}},
       &slice_params,
       sizeof(slice_params)});
}

} // namespace habana
