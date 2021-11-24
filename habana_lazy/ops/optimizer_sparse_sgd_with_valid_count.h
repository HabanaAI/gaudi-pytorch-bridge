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
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace ir {

class OptimizerSparseSgdValidCount : public Node {
 public:
  enum class OptSgdIndex { kMomIdx = 6, kNesterov };
  OptimizerSparseSgdValidCount() = delete;
  OptimizerSparseSgdValidCount(
      const at::Tensor& gradients,
      const at::Tensor& weights_in,
      const at::Tensor& moments_in,
      const at::Tensor& indices,
      const at::Tensor& learning_rate,
      const at::Tensor& valid_count_tensor,
      float mom,
      bool nesterov)
      : ir::Node(c10::Symbol::fromQualString("hpu::habanaOptimizerSparseSgd")) {
    auto hl_gradient = GetOrCreateHbLazyTensor(gradients, c10::kHPU);
    auto hl_wt = GetOrCreateHbLazyTensor(weights_in, c10::kHPU);
    auto hl_mom = GetOrCreateHbLazyTensor(moments_in, c10::kHPU);
    auto hl_indices = GetOrCreateHbLazyTensor(indices, c10::kHPU);
    auto hl_lr = GetOrCreateHbLazyTensor(learning_rate, c10::kHPU);
    auto hl_vc = GetOrCreateHbLazyTensor(valid_count_tensor, c10::kHPU);

    hl_gradient = HandleViewsOrUpdate(gradients, hl_gradient);
    hl_wt = HandleViewsOrUpdate(weights_in, hl_wt);
    hl_mom = HandleViewsOrUpdate(moments_in, hl_mom);
    hl_indices = HandleViewsOrUpdate(indices, hl_indices);
    hl_lr = HandleViewsOrUpdate(learning_rate, hl_lr);
    hl_vc = HandleViewsOrUpdate(valid_count_tensor, hl_vc);

    AddInput(hl_gradient.GetIrValue());
    AddInput(hl_wt.GetIrValue());
    AddInput(hl_mom.GetIrValue());
    AddInput(hl_indices.GetIrValue());
    AddInput(hl_lr.GetIrValue());
    AddInput(hl_vc.GetIrValue());

    m_meta_data.set(mom, static_cast<size_t>(OptSgdIndex::kMomIdx));
    m_meta_data.set(nesterov, static_cast<size_t>(OptSgdIndex::kNesterov));
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << Node::ToString() << ", mom="
       << m_meta_data.get(static_cast<size_t>(OptSgdIndex::kMomIdx)).toDouble()
       << ", nesterov="
       << m_meta_data.get(static_cast<size_t>(OptSgdIndex::kNesterov)).toBool();
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
