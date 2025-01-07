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

#include "hpu_ops/bincount.h"

namespace habana {

OutputMetaDataVector BinCountMeta(const at::Stack& stack) {
  int64_t length = stack.at(1).toInt();
  auto weights = stack.at(2).toOptional<at::Tensor>();

  OutputMetaData meta;
  meta.shape = {length};
  meta.dtype = weights.has_value() ? weights.value().scalar_type()
                                   : c10::ScalarType::Int;
  return {meta};
}

BinCount::BinCount(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(BinCountMeta);
}

void BinCount::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(
      !graph.is_dynamic_graph(), "Dynamic graph is not supported for bincount");

  StackGetter stackGetter(this, stack, "Bincount::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  auto length = stackGetter.getNextInput<int32_t>();
  auto weights = stackGetter.getNextInput<c10::optional<TensorsPair>>();

  ns_BinCountKernel::Params params{
      weights.has_value() ? BinCountMode_t::USE_WEIGHT
                          : BinCountMode_t::NO_WEIGHT};
  auto meta = BinCountMeta(stack).at(0);

  auto constantHolder = ConstantHelper(graph, length, c10::ScalarType::Int);
  std::vector<synTensor> inputs{self.syn_t, constantHolder.get()};
  if (weights.has_value()) {
    inputs.push_back(weights.value().syn_t);
  }

  auto bincount = BuildOp(
      graph,
      get_guid_with_precision("bincount", meta.dtype),
      std::move(inputs),
      {{meta.shape, meta.dtype, 0}},
      (void*)&params,
      sizeof(params));

  syn_out(0) = std::move(bincount.at(0));
}

} // namespace habana

static const auto& BinCountKernelRegistry = habana::KernelRegistry().add(
    "hpu::bincount_backend",
    KERNEL_FN_GLOBAL(habana::BinCount));
