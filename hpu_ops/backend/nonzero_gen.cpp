/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "habana_helpers/frontend_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/nonzero.h"

using namespace torch;
using namespace habana;

namespace habana {

OutputMetaDataVector NonzeroMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);

  OutputMetaDataVector meta(2);
  meta.at(0).shape = self.sizes().vec();
  meta.at(0).dtype = c10::ScalarType::Long;
  meta.at(1).shape = {5};
  meta.at(1).dtype = at::ScalarType::Int; // shape tensor
  return meta;
}

NonZeroEager::NonZeroEager(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0, 0}, {}, {}, false) {
  SetOutputMetaFn(NonzeroMeta);
}

static float round_dims(NonZeroParams_t self_params, int group_size) {
  auto group_size_f = static_cast<float>(group_size);
  auto last_dim_rounded =
      std::ceil(
          self_params.sizes[(int)self_params.sizes.size() - 1] / group_size_f) *
      group_size_f;
  return last_dim_rounded;
}

std::vector<int64_t> compute_output_st_shape(NonZeroParams_t self_params) {
  constexpr int group_size = 64;
  auto last_dim_rounded = round_dims(self_params, group_size);
  auto out_st_shape = self_params.sizes;
  auto group_size_aligned_dim =
      (long int)last_dim_rounded / (long int)group_size;
  out_st_shape.pop_back();
  out_st_shape.emplace_back(group_size_aligned_dim);
  out_st_shape.emplace_back(group_size);
  return out_st_shape;
}

std::vector<int64_t> compute_nonzero_output_shape(
    NonZeroParams_t self_params,
    bool use_tpc_impl) {
  auto input_shape = self_params.sizes;
  int dimensions = input_shape.size();
  auto elements = self_params.numel;
  if ((habana::HPURegistrar::get_device().type() !=
       synDeviceType::synDeviceGreco) and
      (dimensions <= 4) and (dimensions > 0) and !use_tpc_impl) {
    elements = 1;
    auto last_dim_rounded = round_dims(self_params, 64);
    for (unsigned i = 0; i < dimensions - 1; i++) {
      elements *= self_params.sizes[i];
    }
    elements = elements * last_dim_rounded;
  }
  std::vector<int64_t> output_shape{elements, dimensions};
  return output_shape;
}

std::vector<synapse_helpers::tensor> NonZeroCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    NonZeroParams_t self_params,
    synTensor self_synin,
    c10::optional<int> final_result_index_0,
    c10::optional<int> final_result_index_1,
    bool use_tpc_impl = false) {
  auto output_shape = compute_nonzero_output_shape(self_params, use_tpc_impl);
  auto shape_tensor_shape = DimVector{5};
  ns_NonzeroV2::Params params = {};
  std::vector<synTensor> inputs = {self_synin};
  auto guid = get_guid_with_precision("non_zero_v2_fwd", self_params.dtype);

  if (self_params.sizes.size() < 5 and not use_tpc_impl) {
    // Need to create a reshape_shape_tensor for nonzero_v2 guid
    auto st_shape = compute_output_st_shape(self_params);
    op->CreateShapeTensorInput(
        graph, c10::ScalarType::Int, st_shape, inputs, SHAPE_TENSOR, true);
    params.group_size = 64;
  }

  // outputs - coordinates tensor is of output_shape with maximum
  // self.numel() * self.dim() shape
  // and shape_tensor is having 5D shape filled by tpc with actual num of
  // nonzero elems
  return OpBackend::BuildNode(
      op,
      graph,
      {guid,
       inputs,
       {{output_shape, c10::ScalarType::Int, final_result_index_0},
        {shape_tensor_shape, c10::ScalarType::Int, final_result_index_1}},
       &params,
       sizeof(params)});
}

void NonZeroEager::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);

  NonZeroParams_t self_params;
  self_params.dtype = self.scalar_type();
  self_params.sizes = self.sizes().vec();
  self_params.numel = self.numel();

  auto nonzero = NonZeroCommon(this, graph, self_params, syn_in(0), 0, 1);
  syn_out(0) = std::move(nonzero.at(0));
  syn_out(1) = std::move(nonzero.at(1));
}
} // namespace habana

static const auto& NonZeroKernelRegistry = habana::KernelRegistry().add(
    "hpu::nonzero_eager",
    KERNEL_FN_GLOBAL(habana::NonZeroEager));
