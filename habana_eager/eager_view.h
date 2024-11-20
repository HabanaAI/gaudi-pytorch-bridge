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
#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/get_n_bytes.h"
#include "habana_eager/eager_exec.h"
#include "habana_kernels/kernel_utils.h"

using SmallTensorVector = c10::SmallVector<at::Tensor, 8>;

namespace habana {
namespace eager {

using JitGraph = torch::jit::Graph;
using JitNode = torch::jit::Node;
using JitValue = torch::jit::Value;

class ViewParam {
 public:
  ViewParam() {
    sizes = {-1};
    strides = {-1};
    offset = -1;
    total_num_elements = -1;
  }

  void setParam(const at::Tensor& t) {
    auto* impl = t.unsafeGetTensorImpl();
    sizes.clear();
    for (int64_t s : impl->sizes()) {
      sizes.emplace_back(s);
    }
    strides.clear();
    for (int64_t s : impl->strides()) {
      strides.emplace_back(s);
    }
    offset = impl->storage_offset();
    int64_t elem_size =
        c10::elementSize(habana_helpers::getInternalDtype(t.scalar_type()));
    total_num_elements = (int64_t)(habana_helpers::GetNBytes(impl) / elem_size);
  }

  const std::vector<int64_t>& getViewSizes() const {
    return sizes;
  }
  const std::vector<int64_t>& getViewStrides() const {
    return strides;
  }
  int64_t getViewOffset() const {
    return offset;
  }
  int64_t getTotalElements() const {
    return total_num_elements;
  }

 private:
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset;
  int64_t total_num_elements;
};

void HandleOutputInsert(
    JitGraph& graph,
    std::vector<at::IValue>& inputs,
    EagerOpMetaData& eager_op_meta_data,
    CValPtrMap& jit_val_map);

void HandleInputOutputViews(
    JitGraph& graph,
    const c10::ArrayRef<at::IValue> inputs,
    EagerOpMetaData& eager_op_meta_data,
    CValPtrMap& jit_val_map);

void set_as_strided_meta(JitNode* node);
void set_deterministic(JitNode* node);

} // namespace eager
} // namespace habana
