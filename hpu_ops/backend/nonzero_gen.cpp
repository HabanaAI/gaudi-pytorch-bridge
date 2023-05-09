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

float round_dims(const at::Tensor& input_tensor, int group_size) {
  auto group_size_f = static_cast<float>(group_size);
  auto last_dim_rounded =
      std::ceil(input_tensor.sizes()[input_tensor.dim() - 1] / group_size_f) *
      group_size_f;
  return last_dim_rounded;
}

std::vector<int64_t> compute_output_st_shape(const at::Tensor& input_tensor) {
  constexpr int group_size = 64;
  auto last_dim_rounded = round_dims(input_tensor, group_size);
  auto out_st_shape = input_tensor.sizes().vec();
  auto group_size_aligned_dim =
      (long int)last_dim_rounded / (long int)group_size;
  out_st_shape.pop_back();
  out_st_shape.emplace_back(group_size_aligned_dim);
  out_st_shape.emplace_back(group_size);
  return out_st_shape;
}

std::vector<int64_t> compute_nonzero_output_shape(const at::Tensor& self) {
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  auto elements = self.numel();
  if ((synapse_helpers::HPURegistrar::get_device().type() !=
       synDeviceType::synDeviceGreco) and
      (self.dim() <= 4) and (self.dim() > 0)) {
    elements = 1;
    auto last_dim_rounded = round_dims(self, 64);
    for (unsigned i = 0; i < self.sizes().size() - 1; i++) {
      elements *= self.sizes()[i];
    }
    elements = elements * last_dim_rounded;
  }
  std::vector<int64_t> output_shape{elements, dimensions};
  return output_shape;
}

void NonZeroEager::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto shape_tensor_shape = DimVector{5};
  std::vector<synTensor> nonzero_synTensor;
  std::vector<synapse_helpers::tensor> nonzero;

  if (self.dim() > 4) {
    auto guid = get_guid_with_precision("non_zero_fwd", self.scalar_type());
    auto output_shape = compute_nonzero_output_shape(self);
    // outputs - coordinates tensor is of output_shape with maximum
    // self.numel()xself.dim() shape
    //  and shape_tensor is having 5D shape filled by tpc with actual num of
    //  nonzero elems
    synDataType synType = syn_type_uint32;
    std::vector<synTensor> inputs = {syn_in(0)};
    nonzero = OpBackend::BuildNode(
        this,
        graph,
        {guid,
         std::move(inputs),
         {{output_shape, c10::ScalarType::Int, 0, DATA_TENSOR},
          {shape_tensor_shape,
           c10::ScalarType::Int,
           c10::nullopt,
           DATA_TENSOR,
           synType}}});
    syn_out(0) = std::move(nonzero.at(0));
    nonzero_synTensor.emplace_back(nonzero.at(1).get());
  } else {
    auto v2_guid =
        get_guid_with_precision("non_zero_v2_fwd", self.scalar_type());
    auto output_shape = compute_nonzero_output_shape(self);
    auto st_shape = compute_output_st_shape(self);
    // Need to create a reshape_shape_tensor for nonzero_v2 guid here
    std::vector<synTensor> inputs = {syn_in(0)};
    CreateShapeTensorInput(
        graph, c10::ScalarType::Int, st_shape, inputs, SHAPE_TENSOR, true);
    // outputs - coordinates tensor is of output_shape with maximum
    // self.numel()xself.dim() shape
    //  and shape_tensor is having 5D shape filled by tpc with actual num of
    //  nonzero elems
    ns_NonzeroV2::Params params = {};
    params.group_size = 64;
    synDataType synType = syn_type_uint32;
    nonzero = OpBackend::BuildNode(
        this,
        graph,
        {v2_guid,
         std::move(inputs),
         {{output_shape, c10::ScalarType::Int, 0, DATA_TENSOR},
          {shape_tensor_shape,
           c10::ScalarType::Int,
           c10::nullopt,
           DATA_TENSOR,
           synType}},
         &params,
         sizeof(params)});
    syn_out(0) = std::move(nonzero.at(0));
    nonzero_synTensor.emplace_back(nonzero.at(1).get());
  }
  // Add cast for second syn_out - this has to be of type syn_type_uint32
  auto cast_out_shape = OpBackend::BuildNode(
      this,
      graph,
      {"cast_u32_to_i32",
       nonzero_synTensor,
       {{shape_tensor_shape, c10::ScalarType::Int, 1}}});
  syn_out(1) = std::move(cast_out_shape.at(0));
}
} // namespace habana

static const auto& NonZeroKernelRegistry = habana::KernelRegistry().add(
    "hpu::nonzero_eager",
    KERNEL_FN_GLOBAL(habana::NonZeroEager));
