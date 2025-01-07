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
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace ir {

class AsStridedLayout : public ir::Node {
 public:
  enum class AsStridedLayoutIdx { kDimIdx = 1 };
  AsStridedLayout() = delete;
  AsStridedLayout(
      const at::Tensor& self,
      at::IntArrayRef dims,
      std::string op = "aten::permute")
      : Node(c10::Symbol::fromQualString(op)) {
    HbLazyTensor hl_self = GetHbLazyTensor(self);

    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dims, static_cast<size_t>(AsStridedLayoutIdx::kDimIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dims="
       << m_meta_data.get(static_cast<size_t>(AsStridedLayoutIdx::kDimIdx));
    return ss.str();
  }
};

class Transpose : public ir::Node {
 public:
  enum class TransposeIdx { kDim0Idx = 1, kDim1Idx = 2 };
  Transpose() = delete;
  Transpose(const at::Tensor& self, int64_t dim0, int64_t dim1)
      : Node(c10::Symbol::fromQualString("aten::transpose")) {
    HbLazyTensor hl_self = GetHbLazyTensor(self);

    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim0, static_cast<size_t>(TransposeIdx::kDim0Idx));
    m_meta_data.set(dim1, static_cast<size_t>(TransposeIdx::kDim1Idx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim0="
       << m_meta_data.get(static_cast<size_t>(TransposeIdx::kDim0Idx))
       << ", dim1="
       << m_meta_data.get(static_cast<size_t>(TransposeIdx::kDim1Idx));
    return ss.str();
  }
};

class Permute : public ir::Node {
 public:
  enum class PermuteIdx { kDimIdx = 1 };
  Permute() = delete;
  Permute(
      const at::Tensor& self,
      at::IntArrayRef dims,
      std::string op = "aten::permute")
      : Node(c10::Symbol::fromQualString(op)) {
    HbLazyTensor hl_self = GetHbLazyTensor(self);

    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dims, static_cast<size_t>(PermuteIdx::kDimIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dims="
       << m_meta_data.get(static_cast<size_t>(PermuteIdx::kDimIdx));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
