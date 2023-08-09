/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "habana_helpers/logging.h"
#include "hpu_ops/instance_norm.h"

namespace habana {

OutputMetaDataVector InstanceNorm::InstanceNormMeta(const at::Stack& stack) {
  constexpr size_t INPUT_BATCH_INDEX = 0;
  constexpr size_t INPUT_CHANNEL_INDEX = 1;

  OutputMetaDataVector meta(3);
  const auto& input = stack.at(0).toTensor();

  meta.at(0).shape = input.sizes().vec();
  meta.at(1).shape = {
      input.sizes().vec()[INPUT_BATCH_INDEX],
      input.sizes().vec()[INPUT_CHANNEL_INDEX]};
  meta.at(2).shape = {
      input.sizes().vec()[INPUT_BATCH_INDEX],
      input.sizes().vec()[INPUT_CHANNEL_INDEX]};

  meta.at(0).dtype = input.scalar_type();
  meta.at(1).dtype = c10::ScalarType::Float;
  meta.at(2).dtype = c10::ScalarType::Float;
  return meta;
}

InstanceNorm::InstanceNorm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "instance_norm_fwd",
          scalar_type,
          {0, 0, 0},
          {},
          {},
          false) {
  SetOutputMetaFn(InstanceNormMeta);
}

void InstanceNorm::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = InstanceNormMeta(stack);

  TORCH_CHECK(stack[3].isDouble(), "Input type expected to be double");

  auto input = stack[0].toTensor();
  auto is_norm_3d = input.sizes().vec().size() == 5;

  kernel_meta_data_.synapse_input_layout.assign(
      {is_norm_3d ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
                  : synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
  kernel_meta_data_.synapse_output_layout.assign(
      {is_norm_3d ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
                  : synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::CN,
       synapse_helpers::layouts::SynapseLayoutFormat::CN});

  std::string guid =
      get_guid_with_precision("instance_norm_fwd", input.scalar_type());

  const auto eps = stack[3].toDouble();

  // Note: TPC kernel doesnt support running mean and variance computation. we
  // just pass random momentum value as a place holder
  struct ns_InstanceNormTrainingKernel::Params params {
    0.9, static_cast<float>(eps)
  };
  auto instanceNorm = BuildOp(
      graph,
      guid,
      {syn_in(0), syn_in(2), syn_in(1)},
      {{meta[0].shape, meta[0].dtype, 0},
       {meta[1].shape, c10::ScalarType::Float, 1},
       {meta[2].shape, c10::ScalarType::Float, 2}},
      &params,
      sizeof(params));

  syn_out(0) = std::move(instanceNorm[0]);
  syn_out(1) = std::move(instanceNorm[1]);
  syn_out(2) = std::move(instanceNorm[2]);
}

} // namespace habana

static const auto& InstanceNormForwardKernelRegistry =
    habana::KernelRegistry().add(
        "hpu::instance_norm",
        KERNEL_FN_GLOBAL(habana::InstanceNorm));
