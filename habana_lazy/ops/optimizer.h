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

class OptimizerFusedAdagrad : public Node {
 public:
  enum class OptFusedAdaIndex { kwdIdx = 5, klrdIdx, kepsIdx };
  OptimizerFusedAdagrad() = delete;
  OptimizerFusedAdagrad(
      const TensorList& gradients,
      TensorList& weights,
      TensorList& variances,
      const at::Tensor& epoch_num,
      const at::Tensor& lr,
      const float wd,
      const float lrd,
      const float epsilon)
      : ir::Node(
            c10::Symbol::fromQualString("hpu::habanaOptimizerFusedAdagrad")) {
    AddInputVec(gradients);
    AddInputVec(weights);
    AddInputVec(variances);

    auto hl_epoch_num = GetOrCreateHbLazyTensor(epoch_num, c10::kHABANA);
    AddInput(hl_epoch_num.GetIrValue());

    auto hl_lr = GetOrCreateHbLazyTensor(lr, c10::kHABANA);
    AddInput(hl_lr.GetIrValue());

    m_meta_data.set(wd, static_cast<size_t>(OptFusedAdaIndex::kwdIdx));
    m_meta_data.set(lrd, static_cast<size_t>(OptFusedAdaIndex::klrdIdx));
    m_meta_data.set(epsilon, static_cast<size_t>(OptFusedAdaIndex::kepsIdx));
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << Node::ToString() << ", wd="
       << m_meta_data.get(static_cast<size_t>(OptFusedAdaIndex::kwdIdx))
              .toDouble()
       << ", lrd="
       << m_meta_data.get(static_cast<size_t>(OptFusedAdaIndex::klrdIdx))
              .toDouble()
       << ", eps="
       << m_meta_data.get(static_cast<size_t>(OptFusedAdaIndex::kepsIdx))
              .toDouble();

    return ss.str();
  }

 private:
  void AddInputVec(const TensorList& tensor_list) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& t : tensor_list) {
      auto hl_tensor = GetOrCreateHbLazyTensor(t, c10::kHABANA);
      hl_tensors.push_back(hl_tensor.GetIrValue());
      input_pt_vec.emplace_back(t);
    }

    auto input = GetIrValueForListConstruct(hl_tensors);
    input.mp_node->AddInputPtTensors(input_pt_vec);
    AddInput(input);
  }
};

}; // namespace ir
}; // namespace habana_lazy
