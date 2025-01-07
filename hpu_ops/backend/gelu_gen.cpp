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
#include "generated/backend/gelu.h"
#include "generated/backend/gelu_backward.h"
#include "hpu_ops/op_backend.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {
OutputMetaDataVector GeluMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  meta.mem_format = self.suggest_memory_format();
  return {meta};
}

SharedMetaDataVector GeluSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return Input0ToOut0And1SharedMeta(stack, "gelu_fwd");
}

std::shared_ptr<void> FillGeluParams(
    const at::Stack& stack,
    size_t& size,
    int approx_index) {
  PARAMS_STUB(ns_GeluKernel::Params);
  if (GET_ENV_FLAG_NEW(PT_HPU_FORCE_TANH_FOR_GELU)) {
    params->approximation = true;
    return params;
  } else {
    params->approximation = stack.at(approx_index).to<std::string>() == "tanh";
    return params;
  }
}

std::shared_ptr<void> FillGeluFwdParams(const at::Stack& stack, size_t& size) {
  return FillGeluParams(stack, size, 1 /*Approximation Index in Fwd pass*/);
}

std::shared_ptr<void> FillGeluBwdParams(const at::Stack& stack, size_t& size) {
  return FillGeluParams(stack, size, 2 /*Approximation Index in Bwd pass*/);
}

void Gelu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto meta = GeluMeta(stack)[0];
  size_t size = 0;
  auto params = FillGeluFwdParams(stack, size);
  auto gelu = BuildOp(
      graph,
      get_guid_with_precision("gelu_fwd", meta.dtype),
      {syn_in(0)},
      {{meta.shape, meta.dtype, 0}, {meta.shape, meta.dtype}},
      params.get(),
      size);
  syn_out(0) = std::move(gelu[0]);
}

} // namespace habana
