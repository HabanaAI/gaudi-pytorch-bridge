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
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace ir {

struct MseLoss : public habana_lazy::ir::Node {
  MseLoss() = delete;
  MseLoss(const at::Tensor& self, const at::Tensor& target, int64_t reduction)
      : Node(c10::Symbol::fromQualString("aten::mse_loss")),
        m_reduction_index{2} {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto hl_target = habana_lazy::GetOrCreateHbLazyTensor(target, c10::kHABANA);

    AddInput(hl_self.GetIrValue());
    AddInput(hl_target.GetIrValue());

    m_meta_data.set(reduction, m_reduction_index);
  }

  MseLoss(
      const at::Tensor& grad_output,
      const at::Tensor& self,
      const at::Tensor& target,
      int64_t reduction)
      : Node(c10::Symbol::fromQualString("aten::mse_loss_backward")),
        m_reduction_index{3} {
    auto hl_grad_output =
        habana_lazy::GetOrCreateHbLazyTensor(grad_output, c10::kHABANA);
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto hl_target = habana_lazy::GetOrCreateHbLazyTensor(target, c10::kHABANA);

    AddInput(hl_grad_output.GetIrValue());
    AddInput(hl_self.GetIrValue());
    AddInput(hl_target.GetIrValue());

    m_meta_data.set(reduction, m_reduction_index);
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString()
       << ", reduction=" << m_meta_data.get(m_reduction_index);
    return ss.str();
  }

 private:
  const int m_reduction_index;
};
} // namespace ir
} // namespace habana_lazy
