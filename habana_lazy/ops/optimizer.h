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

class OptimizerFusedAdamw : public Node {
 public:
  enum class OptimizerFusedAdamwIndex {
    kbeta1Idx = 6,
    kbeta2Idx,
    kepsIdx,
    kwdIdx
  };
  OptimizerFusedAdamw() = delete;
  OptimizerFusedAdamw(
      const TensorList& gradients,
      TensorList& weights,
      TensorList& exp_avg,
      TensorList& exp_avg_sq,
      at::Tensor& lr_t,
      at::Tensor& neg_step_t,
      const float beta1,
      const float beta2,
      const float epsilon,
      const float weight_decay)
      : ir::Node(c10::Symbol::fromQualString("hpu::habanaOptimizerAdamW")) {
    AddInputVec(gradients);
    AddInputVec(weights);
    AddInputVec(exp_avg);
    AddInputVec(exp_avg_sq);

    auto hl_lr_t = GetOrCreateHbLazyTensor(lr_t, c10::kHABANA);
    AddInput(hl_lr_t.GetIrValue());

    auto hl_neg_step_t = GetOrCreateHbLazyTensor(neg_step_t, c10::kHABANA);
    AddInput(hl_neg_step_t.GetIrValue());

    m_meta_data.set(
        beta1, static_cast<size_t>(OptimizerFusedAdamwIndex::kbeta1Idx));
    m_meta_data.set(
        beta2, static_cast<size_t>(OptimizerFusedAdamwIndex::kbeta2Idx));
    m_meta_data.set(
        epsilon, static_cast<size_t>(OptimizerFusedAdamwIndex::kepsIdx));
    m_meta_data.set(
        weight_decay, static_cast<size_t>(OptimizerFusedAdamwIndex::kwdIdx));
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
       << ", weight_decay="
       << m_meta_data.get(static_cast<size_t>(OptimizerFusedAdamwIndex::kwdIdx))
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

class OptimizerFusedSGD : public Node {
 public:
  enum class OptFusedSGDIndex { kwdIdx = 3, kmomIdx, kdampIdx, knesterovIdx };
  OptimizerFusedSGD() = delete;
  OptimizerFusedSGD(
      const TensorList& gradients,
      TensorList& weights,
      const at::Tensor& lr,
      const float wd,
      const float mom,
      const float damp,
      const bool nesterov)
      : ir::Node(c10::Symbol::fromQualString("hpu::habanaOptimizerFusedSGD")) {
    AddInputVec(gradients);
    AddInputVec(weights);

    auto hl_lr = GetOrCreateHbLazyTensor(lr, c10::kHABANA);
    AddInput(hl_lr.GetIrValue());

    m_meta_data.set(wd, static_cast<size_t>(OptFusedSGDIndex::kwdIdx));
    m_meta_data.set(mom, static_cast<size_t>(OptFusedSGDIndex::kmomIdx));
    m_meta_data.set(damp, static_cast<size_t>(OptFusedSGDIndex::kdampIdx));
    m_meta_data.set(
        nesterov, static_cast<size_t>(OptFusedSGDIndex::knesterovIdx));
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << Node::ToString() << ", wd="
       << m_meta_data.get(static_cast<size_t>(OptFusedSGDIndex::kwdIdx))
              .toDouble()
       << ", momentum="
       << m_meta_data.get(static_cast<size_t>(OptFusedSGDIndex::kmomIdx))
              .toDouble()
       << ", dampening="
       << m_meta_data.get(static_cast<size_t>(OptFusedSGDIndex::kdampIdx))
              .toDouble()
       << ", nesterov="
       << m_meta_data.get(static_cast<size_t>(OptFusedSGDIndex::knesterovIdx))
              .toBool();

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

class OptimizerFusedSGDMomentum : public Node {
 public:
  enum class OptFusedSGDMomentumIndex {
    kwdIdx = 5,
    kmomIdx,
    kdampIdx,
    knesterovIdx
  };
  OptimizerFusedSGDMomentum() = delete;
  OptimizerFusedSGDMomentum(
      const TensorList& gradients,
      TensorList& weights,
      TensorList& momentum,
      const at::Tensor& epoch_num,
      const at::Tensor& lr,
      const float wd,
      const float mom,
      const float damp,
      const bool nesterov)
      : ir::Node(c10::Symbol::fromQualString(
            "hpu::habanaOptimizerFusedSGDMomentum")) {
    AddInputVec(gradients);
    AddInputVec(weights);
    AddInputVec(momentum);

    auto hl_epoch_num = GetOrCreateHbLazyTensor(epoch_num, c10::kHABANA);
    AddInput(hl_epoch_num.GetIrValue());

    auto hl_lr = GetOrCreateHbLazyTensor(lr, c10::kHABANA);
    AddInput(hl_lr.GetIrValue());

    m_meta_data.set(wd, static_cast<size_t>(OptFusedSGDMomentumIndex::kwdIdx));
    m_meta_data.set(
        mom, static_cast<size_t>(OptFusedSGDMomentumIndex::kmomIdx));
    m_meta_data.set(
        damp, static_cast<size_t>(OptFusedSGDMomentumIndex::kdampIdx));
    m_meta_data.set(
        nesterov, static_cast<size_t>(OptFusedSGDMomentumIndex::knesterovIdx));
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << Node::ToString() << ", wd="
       << m_meta_data.get(static_cast<size_t>(OptFusedSGDMomentumIndex::kwdIdx))
              .toDouble()
       << ", momentum="
       << m_meta_data
              .get(static_cast<size_t>(OptFusedSGDMomentumIndex::kmomIdx))
              .toDouble()
       << ", dampening="
       << m_meta_data
              .get(static_cast<size_t>(OptFusedSGDMomentumIndex::kdampIdx))
              .toDouble()
       << ", nesterov="
       << m_meta_data
              .get(static_cast<size_t>(OptFusedSGDMomentumIndex::knesterovIdx))
              .toBool();

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
