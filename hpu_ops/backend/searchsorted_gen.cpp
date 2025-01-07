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
#include "generated/backend/searchsorted.h"

namespace habana {
OutputMetaDataVector SearchSortedMeta(const at::Stack& stack) {
  std::vector<int64_t> outshape;
  if (stack.at(1).isTensor()) {
    auto self = stack_tensor(stack, 1);
    outshape = self.sizes().vec();
  } else {
    outshape = {1};
  }
  auto sortedseq = stack_tensor(stack, 0);
  auto seqshape = sortedseq.sizes().vec();

  auto old_seqshape = seqshape;
  auto old_outshape = outshape;
  old_seqshape.erase(old_seqshape.end() - 1);
  old_outshape.erase(old_outshape.end() - 1);
  TORCH_CHECK(
      seqshape.empty() || (old_seqshape == old_outshape),
      "torch.searchsorted(): boundaries tensor should be 1 dimension or ",
      "the first N-1 dimensions of boundaries tensor and input value tensor ",
      "must match, but we got boundaries tensor ",
      seqshape,
      "and input value tensor ",
      outshape);
  bool out_int32 = stack.at(2).toBool();
  OutputMetaData meta;
  meta.shape = outshape;
  meta.dtype = out_int32 ? torch::kInt32 : torch::kLong;
  return {meta};
}

std::shared_ptr<void> FillSearchSortedParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_SearchSorted::Params);
  bool right = stack.at(3).toBool();
  if (stack.at(4).isString()) {
    right = stack.at(4).toStringView() == "right";
  }

  params->right = right;
  return params;
}

} // namespace habana