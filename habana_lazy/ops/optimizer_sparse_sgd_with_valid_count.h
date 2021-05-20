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
    std::vector<HbLazyTensor> hl_tensors;
    hl_tensors.push_back(GetOrCreateHbLazyTensor(gradients, c10::kHABANA));
    hl_tensors.push_back(GetOrCreateHbLazyTensor(weights_in, c10::kHABANA));
    hl_tensors.push_back(GetOrCreateHbLazyTensor(moments_in, c10::kHABANA));
    hl_tensors.push_back(GetOrCreateHbLazyTensor(indices, c10::kHABANA));
    hl_tensors.push_back(GetOrCreateHbLazyTensor(learning_rate, c10::kHABANA));
    hl_tensors.push_back(
        GetOrCreateHbLazyTensor(valid_count_tensor, c10::kHABANA));

    for (auto& i : hl_tensors) {
      AddInput(i.GetIrValue());
    }

    std::vector<at::Tensor> input_pt_vec{
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor};
    AddInputPtTensors(input_pt_vec);

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
