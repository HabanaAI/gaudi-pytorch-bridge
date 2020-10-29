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

class View : public ir::Node {
 public:
  enum class ViewMeta {
    WEIGHT_INDEX = 1,
    BIAS_INDEX,
    M_INDEX,
    N_INDEX,
    EPS_INDEX
  };
  View() = delete;
  View(const Tensor& self, IntArrayRef size)
      : Node(c10::Symbol::fromQualString("aten::view")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_self.GetIrValue());
    m_meta_data.set(size, static_cast<size_t>(1));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString()
       << ", View Size = " << m_meta_data.get(static_cast<size_t>(1));
    return ss.str();
  }
};
} // namespace ir
} // namespace habana_lazy
