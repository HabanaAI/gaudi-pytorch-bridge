
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

#include "backend/habana_operator.h"
#include "hpu_ops/repeat_interleave.h"

namespace habana {

OutputMetaDataVector RepeatInterleaveMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto output_size_opt = stack.at(1).toOptional<int64_t>();
  TORCH_CHECK(
      output_size_opt.has_value(),
      "It is expected that output_size is provided after frontend execution.");

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = std::vector<int64_t>{output_size_opt.value()};

  return {meta};
}

RepeatInterleave::RepeatInterleave(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(RepeatInterleaveMeta);
}

void RepeatInterleave::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto meta = RepeatInterleaveMeta(stack)[0];
  const bool isCastNeeded =
      meta.dtype == torch::kLong && common::IsInt64Supported();

  const auto dtype = torch::kInt;

  TORCH_CHECK(self.dim() == 1, "Self tensor is expected to be 1D.");

  const auto self_numel = self.numel();

  ns_RangeKernel::Params paramsRange{};
    paramsRange.start.i = 0;
    paramsRange.limit.i = static_cast<int>(self_numel);
    paramsRange.delta.i = 1;

    auto range = BuildOp(
        graph,
        get_guid_with_precision("range", torch::kInt),
        {},
        {{self_numel, torch::kInt}},
        &paramsRange,
        sizeof(paramsRange));

    std::unique_ptr<synapse_helpers::tensor> cast;
    if (isCastNeeded) {
      cast = std::make_unique<synapse_helpers::tensor>(OpBackend::BuildCast(
          this, graph, syn_in(0), self.sizes(), meta.dtype, dtype));
    }

    std::vector<synTensor> syn_inputs{
        range[0].get(), isCastNeeded ? cast->get() : syn_in(0)};

    ns_RepeatKernelGaudiTF::Params params{};
    params.axis = 0; // Self is always 1D so axis is 0

    auto result = BuildOp(
        graph,
        get_guid_with_precision("repeat_fwd", dtype),
        std::move(syn_inputs),
        {{meta.shape,
          dtype,
          isCastNeeded ? std::nullopt : std::optional<int>(0)}},
        &params,
        sizeof(params));

    if (isCastNeeded) {
      auto res = OpBackend::BuildCast(
          this, graph, result[0].get(), meta.shape, dtype, meta.dtype, 0);
      syn_out(0) = std::move(res);
    } else {
      syn_out(0) = std::move(result[0]);
    }
}
} // namespace habana

static const auto& HabanaRepeatInterleaveKernelRegistry =
    habana::KernelRegistry().add(
        "aten::repeat_interleave.Tensor",
        KERNEL_FN_GLOBAL(habana::RepeatInterleave));