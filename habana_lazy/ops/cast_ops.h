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
class Cast : public Node {
 public:
  Cast() = delete;
  Cast(const Tensor& self, const Tensor& src, bool non_blocking)
      : Node(c10::Symbol::fromQualString("hpu::cast")) {
    auto hl_src = GetOrCreateHbLazyTensor(src, c10::kHABANA);
    auto ir_value_src = hl_src.GetIrValue();
    AddInput(ir_value_src);
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto ir_value_self = hl_self.GetIrValue();
    AddInput(ir_value_self);
    std::vector<at::Tensor> input_pt_vec{src, self};
    AddInputPtTensors(input_pt_vec);
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString();
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy