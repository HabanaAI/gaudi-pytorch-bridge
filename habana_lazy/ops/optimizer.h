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
    kbeta1Idx = 6,
    kbeta2Idx,
    kepsIdx,
    kwdIdx = 10
  };
  OptimizerFusedAdamw() = delete;
  OptimizerFusedAdamw(
      const at::TensorList& gradients,
      at::TensorList& weights,
      at::TensorList& exp_avg,
      at::TensorList& exp_avg_sq,
      at::Tensor& lr_t,
      at::Tensor& neg_step_t,
      const float beta1,
      const float beta2,
      const float epsilon,
      at::Tensor& weight_decay_t,
      const bool is_wd_modified)
      : ir::Node(c10::Symbol::fromQualString("hpu::habanaOptimizerAdamW")) {
    AddInputVec(gradients);
    AddInputVec(weights);
    AddInputVec(exp_avg);
    AddInputVec(exp_avg_sq);

    auto hl_lr_t = GetOrCreateHbLazyTensor(lr_t, c10::kHPU);
    AddInput(hl_lr_t.GetIrValue());

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

class OptimizerFusedLambPhase1 : public Node {
 public:
  enum class OptimizerFusedLambPhase1Meta {
    kbeta1Idx = 5,
    kbeta2Idx = 6,
    kbeta3Idx = 7,
    kepsIdx = 8,
    kwdIdx = 11
  };
  OptimizerFusedLambPhase1() = delete;
  OptimizerFusedLambPhase1(
      const std::vector<at::Tensor>& gradients,
      std::vector<at::Tensor>& weights,
      std::vector<at::Tensor>& exp_avg,
      std::vector<at::Tensor>& exp_avg_sq,
      const at::Tensor& clip_global_grad_norm,
      const float beta1,
      const float beta2,
      const float beta3,
      const float epsilon,
      at::Tensor& bias_correction1_t,
      at::Tensor& bias_correction2_t,
      const float weight_decay)
      : ir::Node(
            c10::Symbol::fromQualString("hpu::habanaOptimizerLambPhase1")) {
    AddInputVec(gradients);
    AddInputVec(weights);
    AddInputVec(exp_avg);
    AddInputVec(exp_avg_sq);

    auto hl_clip_global_grad_norm =
        GetOrCreateHbLazyTensor(clip_global_grad_norm, c10::kHPU);
    AddInput(hl_clip_global_grad_norm.GetIrValue());

    auto hl_bias_correction1_t =
        GetOrCreateHbLazyTensor(bias_correction1_t, c10::kHPU);
    AddInput(hl_bias_correction1_t.GetIrValue());

    auto hl_bias_correction2_t =
        GetOrCreateHbLazyTensor(bias_correction2_t, c10::kHPU);
    AddInput(hl_bias_correction2_t.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{
        clip_global_grad_norm, bias_correction1_t, bias_correction2_t};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        beta1, static_cast<size_t>(OptimizerFusedLambPhase1Meta::kbeta1Idx));
    m_meta_data.set(
        beta2, static_cast<size_t>(OptimizerFusedLambPhase1Meta::kbeta2Idx));
    m_meta_data.set(
        beta3, static_cast<size_t>(OptimizerFusedLambPhase1Meta::kbeta3Idx));
    m_meta_data.set(
        epsilon, static_cast<size_t>(OptimizerFusedLambPhase1Meta::kepsIdx));
    m_meta_data.set(
        weight_decay,
        static_cast<size_t>(OptimizerFusedLambPhase1Meta::kwdIdx));
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << Node::ToString() << ", beta1="
       << m_meta_data
              .get(static_cast<size_t>(OptimizerFusedLambPhase1Meta::kbeta1Idx))
              .toDouble()
       << ", beta2="
       << m_meta_data
              .get(static_cast<size_t>(OptimizerFusedLambPhase1Meta::kbeta2Idx))
              .toDouble()
       << ", eps="
       << m_meta_data
              .get(static_cast<size_t>(OptimizerFusedLambPhase1Meta::kepsIdx))
              .toDouble()
       << ", weight_decay="
       << m_meta_data
              .get(static_cast<size_t>(OptimizerFusedLambPhase1Meta::kwdIdx))
              .toDouble();

    return ss.str();
  }

 private:
  void AddInputVec(const std::vector<at::Tensor>& tensor_list) {
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
