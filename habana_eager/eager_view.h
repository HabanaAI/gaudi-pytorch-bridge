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
#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include "backend/helpers/get_n_bytes.h"
#include "habana_eager/eager_exec.h"
#include "habana_kernels/kernel_utils.h"
#include "pytorch_helpers/habana_device/hpu_cached_devices.h"

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

  void setParam(at::Tensor& t) {
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

  std::vector<int64_t> getViewSizes() {
    return sizes;
  }
  std::vector<int64_t> getViewStrides() {
    return strides;
  }
  int64_t getViewOffset() {
    return offset;
  }
  int64_t getTotalElements() {
    return total_num_elements;
  }

 private:
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset;
  int64_t total_num_elements;

} __attribute__((aligned(64)));

struct StridedOutInfo {
  size_t index;
  at::Tensor tensor;
  JitValue* value;
  std::unique_ptr<ViewParam> param;
};

void HandleInputOutputViews(
    std::shared_ptr<JitGraph>& graph,
    const std::vector<at::Tensor>& inputs,
    const EagerOpMetaData& eager_op_meta_data);

} // namespace eager
} // namespace habana
