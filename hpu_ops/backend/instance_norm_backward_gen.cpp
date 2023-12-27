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
#include "hpu_ops/instance_norm_backward.h"

namespace habana {
namespace {
constexpr size_t INPUT_CHANNEL_INDEX = 1;
} // namespace

namespace sh = synapse_helpers;

OutputMetaDataVector InstanceNormBackward::InstanceNormBackwardMeta(
    const at::Stack& stack) {
  OutputMetaDataVector meta(3);
  const auto& input = stack.at(0).toTensor();

  meta.at(0).shape = input.sizes().vec();
  meta.at(1).shape = {input.sizes().vec()[INPUT_CHANNEL_INDEX]};
  meta.at(2).shape = {input.sizes().vec()[INPUT_CHANNEL_INDEX]};

  meta.at(0).dtype = input.scalar_type();
  meta.at(1).dtype = c10::ScalarType::Float;
  meta.at(2).dtype = c10::ScalarType::Float;
  return meta;
}

InstanceNormBackward::InstanceNormBackward(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "instance_norm_bwd",
          scalar_type,
          {0, 0, 0},
          {},
          {},
          false) {
  SetOutputMetaFn(InstanceNormBackwardMeta);
}

void InstanceNormBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = InstanceNormBackwardMeta(stack);

  StackGetter stackGetter(stack, "InstanceNormBwd::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto grad_in = getNextInput<TensorsPair>(stackGetter);
  auto mean = getNextInput<TensorsPair>(stackGetter);
  auto istd = getNextInput<TensorsPair>(stackGetter);
  auto gamma = getNextInput<c10::optional<TensorsPair>>(stackGetter);

  auto is_norm_3d = input.pt_t.sizes().vec().size() == 5;

  kernel_meta_data_.synapse_input_layout.assign(
      {is_norm_3d ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
                  : synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       is_norm_3d ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
                  : synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::CN,
       synapse_helpers::layouts::SynapseLayoutFormat::CN,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
  kernel_meta_data_.synapse_output_layout.assign(
      {is_norm_3d ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
                  : synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});

  std::vector<synTensor> inputs = {
      input.syn_t, grad_in.syn_t, mean.syn_t, istd.syn_t};

  std::optional<sh::tensor> gammaTensorStorage = gamma
      ? std::nullopt
      : std::make_optional(ConstantHelper(
            graph,
            1.0f,
            c10::ScalarType::Float,
            input.pt_t.sizes().vec()[INPUT_CHANNEL_INDEX]));

  inputs.push_back(
      gammaTensorStorage.has_value() ? gammaTensorStorage.value().get()
                                     : gamma.value().syn_t);

  // Note: TPC kernel doesnt support running mean and variance computation. we
  // just pass random momentum value as a place holder
  struct ns_InstanceNormTrainingKernel::Params params {
    0.9, static_cast<float>(1e-5)
  };
  auto InstanceNormBackward = BuildOp(
      graph,
      guid_,
      std::move(inputs),
      {{meta[0].shape, meta[0].dtype, 0},
       {meta[1].shape, meta[1].dtype, 1},
       {meta[2].shape, meta[1].dtype, 2}},
      &params,
      sizeof(params));

  syn_out(0) = std::move(InstanceNormBackward[0]);
  syn_out(1) = std::move(InstanceNormBackward[1]);
  syn_out(2) = std::move(InstanceNormBackward[2]);
}

} // namespace habana

static const auto& InstanceNormBackwardKernelRegistry =
    habana::KernelRegistry().add(
        "hpu::instance_norm_backward",
        KERNEL_FN_GLOBAL(habana::InstanceNormBackward));
