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

#include "hpu_ops/custom_softmax.h"

namespace habana {

CustomSoftmax::CustomSoftmax(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "custom_softmax_fwd",
          scalar_type,
          {0},
          {},
          {},
          false) {}

void CustomSoftmax::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "CustomSoftmax::AddNode");
  const auto self = stackGetter.getNextInput<TensorsPair>();
  const auto flavor = stackGetter.getNextInput<int>();

  ns_CustomSoftmax::Params params{};
  params.flavor = flavor;

  if (flavor == 0) {
    SetGuid(get_guid_with_precision("softmax_fwd", ScalarType()));
    // For flavor=0, the classic softmax_fwd is used. In theory
    // it should get ns_Softmax::Params with dim=0, but luckily it has the same
    // size and param type as ns_CustomSoftmax. To make code simpler, especially
    // that in the near future softmax_fwd will be used always,
    // I just pass ns_CustomSoftmax::Params with flavor=0.
  }

  auto output = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       {self.syn_t},
       {{self.pt_t.sizes().vec(), ScalarType(), 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
}
} // namespace habana

static const auto& CustomSoftmaxKernelRegistry = habana::KernelRegistry().add(
    "hpu::custom_softmax",
    KERNEL_FN_GLOBAL(habana::CustomSoftmax));
