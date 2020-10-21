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

}; // namespace ir
}; // namespace habana_lazy