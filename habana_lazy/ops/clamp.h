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

class Clamp : public Node {
 public:
  enum class ClampIndex { kMinIdx = 1, kMaxIdx = 2 };
  Clamp() = delete;
  Clamp(Tensor& self, c10::optional<Scalar> min, c10::optional<Scalar> max)
      : Node(c10::Symbol::fromQualString("aten::clamp")) {
    auto hl_input = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_input.GetIrValue());
    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(min, static_cast<size_t>(ClampIndex::kMinIdx));
    m_meta_data.set(max, static_cast<size_t>(ClampIndex::kMaxIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString()
       << ", min=" << m_meta_data.get(static_cast<size_t>(ClampIndex::kMinIdx))
       << ", max=" << m_meta_data.get(static_cast<size_t>(ClampIndex::kMaxIdx));
    return ss.str();
  }
};
}; // namespace ir

}; // namespace habana_lazy