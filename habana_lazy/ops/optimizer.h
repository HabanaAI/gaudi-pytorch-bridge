/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#pragma once
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/view_utils.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace ir {

class OptimizerFusedAdamw : public Node {
 public:
  enum class OptimizerFusedAdamwIndex {
    kbeta1Idx = 5,
    kbeta2Idx,
    kepsIdx,
    kwdIdx = 9
  };
  OptimizerFusedAdamw() = delete;
  OptimizerFusedAdamw(
      const at::TensorList& gradients,
      at::TensorList& weights,
      at::TensorList& exp_avg,
      at::TensorList& exp_avg_sq,
      const at::Tensor& neg_step_t,
      const float beta1,
      const float beta2,
      const float epsilon,
      const at::Tensor& weight_decay_t,
      const bool is_wd_modified)
      : ir::Node(c10::Symbol::fromQualString("hpu::habanaOptimizerAdamW")) {
    AddInputVec(gradients);
    AddInputVec(weights);
    AddInputVec(exp_avg);
    AddInputVec(exp_avg_sq);

    auto hl_neg_step_t = GetOrCreateHbLazyTensor(neg_step_t, c10::kHPU);
    AddInput(hl_neg_step_t.GetIrValue());

    m_meta_data.set(
        beta1, static_cast<size_t>(OptimizerFusedAdamwIndex::kbeta1Idx));
    m_meta_data.set(
        beta2, static_cast<size_t>(OptimizerFusedAdamwIndex::kbeta2Idx));
    m_meta_data.set(
        epsilon, static_cast<size_t>(OptimizerFusedAdamwIndex::kepsIdx));

    auto hl_weight_decay_t = GetOrCreateHbLazyTensor(weight_decay_t, c10::kHPU);
    AddInput(hl_weight_decay_t.GetIrValue());

    m_meta_data.set(
        is_wd_modified, static_cast<size_t>(OptimizerFusedAdamwIndex::kwdIdx));
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << Node::ToString() << ", beta1="
       << m_meta_data
              .get(static_cast<size_t>(OptimizerFusedAdamwIndex::kbeta1Idx))
              .toDouble()
       << ", beta2="
       << m_meta_data
              .get(static_cast<size_t>(OptimizerFusedAdamwIndex::kbeta2Idx))
              .toDouble()
       << ", eps="
       << m_meta_data
              .get(static_cast<size_t>(OptimizerFusedAdamwIndex::kepsIdx))
              .toDouble()
       << ", is_weight_decay_modified="
       << m_meta_data.get(static_cast<size_t>(OptimizerFusedAdamwIndex::kwdIdx))
              .toBool();

    return ss.str();
  }

 private:
  void AddInputVec(const at::TensorList& tensor_list) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& t : tensor_list) {
      auto hl_tensor = GetOrCreateHbLazyTensor(t, c10::kHPU);
      hl_tensor = HbLazyTensorViews::HandleViewsOrUpdate(t, hl_tensor);
      hl_tensors.push_back(hl_tensor.GetIrValue());
      input_pt_vec.emplace_back(t);
    }

    auto input = GetIrValueForListConstruct(hl_tensors);
    input.mp_node->AddInputPtTensors(input_pt_vec);
    AddInput(input);
  }
};

class OptimizerFusedEMA : public Node {
 public:
  OptimizerFusedEMA() = delete;
  OptimizerFusedEMA(
      const at::TensorList& model_inputs,
      at::TensorList& updated_ema,
      const at::Tensor& decay)
      : ir::Node(c10::Symbol::fromQualString("hpu::habanaOptimizerFusedEMA")) {
    AddInputVec(model_inputs);
    AddInputVec(updated_ema);
    auto hl_decay = GetOrCreateHbLazyTensor(decay, c10::kHPU);
    AddInput(hl_decay.GetIrValue());
  }
  std::string ToString() const {
    std::stringstream ss;
    ss << Node::ToString() << ", decay=";
    return ss.str();
  }

 private:
  void AddInputVec(const at::TensorList& tensor_list) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& t : tensor_list) {
      auto hl_tensor = GetOrCreateHbLazyTensor(t, c10::kHPU);
      hl_tensor = HbLazyTensorViews::HandleViewsOrUpdate(t, hl_tensor);
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
