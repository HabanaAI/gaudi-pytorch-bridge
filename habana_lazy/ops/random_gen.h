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

class Dropout : public Node {
 public:
  enum class DropoutIndex { kPIdx = 1, kGenIdx = 2 };
  Dropout() = delete;
  Dropout(const Tensor& self, double p, CPUGeneratorImpl* gen = nullptr)
      : Node(c10::Symbol::fromQualString("aten::_fused_dropout")) {
    auto hl_input = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_input.GetIrValue());
    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(p, static_cast<size_t>(DropoutIndex::kPIdx));
    //  m_meta_data.set(gen, static_cast<size_t>(DropoutIndex::kGenIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString()
       << ", p=" << m_meta_data.get(static_cast<size_t>(DropoutIndex::kPIdx));
    //     << ", gen=" <<
    //     m_meta_data.get(static_cast<size_t>(DropoutIndex::kGenIdx));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
