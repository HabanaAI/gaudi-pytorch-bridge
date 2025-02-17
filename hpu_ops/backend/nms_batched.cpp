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

#include "hpu_ops/nms_batched.h"

namespace sh = synapse_helpers;

namespace habana {

OutputMetaDataVector ComputeNmsBatchedAlignMetadata(const at::Stack& stack) {
  auto indexes = stack[2].toTensor();
  auto max_classes = stack[4].toScalar().toInt();

  return {
      OutputMetaData(
          c10::ScalarType::Long,
          {static_cast<int>(indexes.sizes()[0]) * max_classes}),
      OutputMetaData(c10::ScalarType::Int, {5})};
}

NmsBatched::NmsBatched(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "batched_nms_fwd",
          scalar_type,
          {0, 0},
          {},
          {},
          false) {
  SetOutputMetaFn(ComputeNmsBatchedAlignMetadata);
}

void NmsBatched::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "NmsBatched::AddNode");
  auto boxes = stackGetter.getNextInput<TensorsPair>();
  auto scores = stackGetter.getNextInput<TensorsPair>();
  auto indexes = stackGetter.getNextInput<TensorsPair>();
  auto iou = stackGetter.getNextInput<double>();
  auto max_classes = stackGetter.getNextInput<int>();
  if (boxes.pt_t.numel() == 0 && scores.pt_t.numel() == 0 &&
      indexes.pt_t.numel() == 0) {
    auto zero_tensor = ConstantHelper(graph, 0, c10::ScalarType::Long, {0}, 0);
    syn_out(0) = std::move(zero_tensor);
    return;
  }

  std::vector<synTensor> inputs = {boxes.syn_t, scores.syn_t, indexes.syn_t};

  std::vector<sh::tensor> storage;
  storage.reserve(2);

  if (boxes.pt_t.scalar_type() == c10::ScalarType::BFloat16) {
    storage.push_back(BuildCast(
        this,
        graph,
        boxes.syn_t,
        boxes.pt_t.sizes().vec(),
        c10::ScalarType::BFloat16,
        c10::ScalarType::Float));
    inputs[0] = storage.back().get();
  }

  if (scores.pt_t.scalar_type() == c10::ScalarType::BFloat16) {
    storage.push_back(BuildCast(
        this,
        graph,
        scores.syn_t,
        scores.pt_t.sizes().vec(),
        c10::ScalarType::BFloat16,
        c10::ScalarType::Float));
    inputs[1] = storage.back().get();
  }

  CreateShapeTensorInput(
      graph,
      c10::ScalarType::Int,
      {indexes.pt_t.sizes()[0]},
      inputs,
      SHAPE_TENSOR,
      true);
  CreateShapeTensorInput(
      graph,
      c10::ScalarType::Int,
      {indexes.pt_t.sizes()[0] * max_classes},
      inputs,
      SHAPE_TENSOR,
      true);

  ns_BatchedNmsKernel::Params params{};
  params.nms_threshold = iou;
  params.max_num_classes = max_classes;

  const auto& output_meta = ComputeNmsBatchedAlignMetadata(stack);
  auto output = BuildNode(
      this,
      graph,
      {update_guid_dtype(guid_, c10::ScalarType::Float),
       std::move(inputs),
       {{output_meta[0].shape, output_meta[0].dtype, 0},
        {output_meta[1].shape, output_meta[1].dtype, 1}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
  syn_out(1) = std::move(output[1]);
}

} // namespace habana

static const auto& NmsBatchedKernelRegistry = habana::KernelRegistry().add(
    "hpu::batched_nms_eager",
    KERNEL_FN_GLOBAL(habana::NmsBatched));
