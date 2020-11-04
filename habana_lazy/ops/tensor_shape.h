/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once
#include "habana_helpers/logging.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace ir {

class Permute : public ir::Node {
 public:
  enum class PermuteIdx { kDimIdx = 1 };
  Permute() = delete;
  Permute(const Tensor& self, IntArrayRef dims)
      : Node(c10::Symbol::fromQualString("aten::permute")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_self.GetIrValue());
    m_meta_data.set(dims, static_cast<size_t>(PermuteIdx::kDimIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dims="
       << m_meta_data.get(static_cast<size_t>(PermuteIdx::kDimIdx));
    return ss.str();
  }
};

class Transpose : public ir::Node {
 public:
  enum class TransposeIdx { kDim0Idx = 1, kDim1Idx = 2 };
  Transpose() = delete;
  Transpose(const Tensor& self, int64_t dim0, int64_t dim1)
      : Node(c10::Symbol::fromQualString("aten::transpose")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_self.GetIrValue());
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

}; // namespace ir
}; // namespace habana_lazy