/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "habana_helpers/logging.h"
#include "hpu_ops/instance_norm.h"

namespace habana {
namespace {
constexpr size_t INPUT_BATCH_INDEX = 0;
constexpr size_t INPUT_CHANNEL_INDEX = 1;
} // namespace

namespace sh = synapse_helpers;

OutputMetaDataVector InstanceNorm::InstanceNormMeta(const at::Stack& stack) {
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

  StackGetter stackGetter(this, stack, "InstanceNormFwd::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto weight = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto bias = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  const auto eps = stackGetter.getNextInput<double>();

  auto is_norm_3d = input.pt_t.sizes().vec().size() == 5;

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

  std::vector<synTensor> inputs = {input.syn_t};

  std::optional<sh::tensor> biasTensorStorage = bias
      ? std::nullopt
      : std::make_optional(ConstantHelper(
            graph,
            0.0f,
            c10::ScalarType::Float,
            input.pt_t.sizes().vec()[INPUT_CHANNEL_INDEX]));

  inputs.push_back(
      biasTensorStorage.has_value() ? biasTensorStorage.value().get()
                                    : bias.value().syn_t);

  std::optional<sh::tensor> weightTensorStorage = weight
      ? std::nullopt
      : std::make_optional(ConstantHelper(
            graph,
            1.0f,
            c10::ScalarType::Float,
            input.pt_t.sizes().vec()[INPUT_CHANNEL_INDEX]));

  inputs.push_back(
      weightTensorStorage.has_value() ? weightTensorStorage.value().get()
                                      : weight.value().syn_t);

  // Note: TPC kernel doesnt support running mean and variance computation. we
  // just pass random momentum value as a place holder
  struct ns_InstanceNormTrainingKernel::Params params {
    0.9, static_cast<float>(eps)
  };
  auto instanceNorm = BuildOp(
      graph,
      guid_,
      std::move(inputs),
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
