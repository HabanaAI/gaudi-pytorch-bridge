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
      const at::TensorList& gradients,
      at::TensorList& weights,
      at::TensorList& variances,
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

    auto hl_epoch_num = GetOrCreateHbLazyTensor(epoch_num, c10::kHPU);
    AddInput(hl_epoch_num.GetIrValue());

    auto hl_lr = GetOrCreateHbLazyTensor(lr, c10::kHPU);
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
  void AddInputVec(const at::TensorList& tensor_list) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& t : tensor_list) {
      auto hl_tensor = GetOrCreateHbLazyTensor(t, c10::kHPU);
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
      const at::TensorList& gradients,
      at::TensorList& weights,
      at::TensorList& exp_avg,
      at::TensorList& exp_avg_sq,
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
  void AddInputVec(const at::TensorList& tensor_list) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& t : tensor_list) {
      auto hl_tensor = GetOrCreateHbLazyTensor(t, c10::kHPU);
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
      const at::TensorList& gradients,
      at::TensorList& weights,
      const at::Tensor& lr,
      const float wd,
      const float mom,
      const float damp,
      const bool nesterov)
      : ir::Node(c10::Symbol::fromQualString("hpu::habanaOptimizerFusedSGD")) {
    AddInputVec(gradients);
    AddInputVec(weights);

    auto hl_lr = GetOrCreateHbLazyTensor(lr, c10::kHPU);
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
  void AddInputVec(const at::TensorList& tensor_list) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& t : tensor_list) {
      auto hl_tensor = GetOrCreateHbLazyTensor(t, c10::kHPU);
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
      const at::TensorList& gradients,
      at::TensorList& weights,
      at::TensorList& momentum,
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

    auto hl_epoch_num = GetOrCreateHbLazyTensor(epoch_num, c10::kHPU);
    AddInput(hl_epoch_num.GetIrValue());

    auto hl_lr = GetOrCreateHbLazyTensor(lr, c10::kHPU);
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
  void AddInputVec(const at::TensorList& tensor_list) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& t : tensor_list) {
      auto hl_tensor = GetOrCreateHbLazyTensor(t, c10::kHPU);
      hl_tensors.push_back(hl_tensor.GetIrValue());
      input_pt_vec.emplace_back(t);
    }

    auto input = GetIrValueForListConstruct(hl_tensors);
    input.mp_node->AddInputPtTensors(input_pt_vec);
    AddInput(input);
  }
};

class LambFusedNorm : public ir::Node {
 public:
  enum class LambFusedNormMeta {
    MAX_GRAD_NORM_INDEX = 1,
  };
  LambFusedNorm() = delete;
  LambFusedNorm(
      const std::vector<at::Tensor>& grad,
      float max_grad_norm,
      at::Tensor& clip_norm)
      : Node(c10::Symbol::fromQualString("hpu::habanaOptimizerLambFusedNorm")) {
    AddInputVec(grad);

    auto hl_clip_norm = GetOrCreateHbLazyTensor(clip_norm, c10::kHPU);
    auto clip_ir = hl_clip_norm.GetIrValue();
    AddInput(clip_ir);

    m_meta_data.set(
        max_grad_norm,
        static_cast<size_t>(LambFusedNormMeta::MAX_GRAD_NORM_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", max_grad_norm = "
       << m_meta_data.get(
              static_cast<size_t>(LambFusedNormMeta::MAX_GRAD_NORM_INDEX));
    return ss.str();
  }

 private:
  void AddInputVec(const std::vector<at::Tensor>& tensor_list) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& t : tensor_list) {
      auto hl_tensor = GetOrCreateHbLazyTensor(t, c10::kHPU);
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

    // Added tensors corresponding to tensorlist just as placeholder inorder to
    // get correct index for input_pt_vec
    std::vector<at::Tensor> input_pt_vec{
        gradients[0],
        weights[0],
        exp_avg[0],
        exp_avg_sq[0],
        clip_global_grad_norm,
        bias_correction1_t,
        bias_correction2_t};
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
      hl_tensors.push_back(hl_tensor.GetIrValue());
      input_pt_vec.emplace_back(t);
    }

    auto input = GetIrValueForListConstruct(hl_tensors);
    input.mp_node->AddInputPtTensors(input_pt_vec);
    AddInput(input);
  }
};

class OptimizerFusedLambPhase2 : public Node {
 public:
  enum class OptimizerFusedLambPhase2Meta { kwdIdx = 6, kuselambIdx = 7 };
  OptimizerFusedLambPhase2() = delete;
  OptimizerFusedLambPhase2(
      std::vector<at::Tensor>& weight_vec,
      const std::vector<at::Tensor>& adam_norm_vec,
      const std::vector<at::Tensor>& weight_norm_vec,
      const std::vector<at::Tensor>& adam_step_vec,
      const std::vector<at::Tensor>& trust_ratio_vec,
      const at::Tensor& neg_step_t,
      const float weight_decay,
      const int use_lamb)
      : ir::Node(
            c10::Symbol::fromQualString("hpu::habanaOptimizerLambPhase2")) {
    AddInputVec(weight_vec);
    AddInputVec(adam_norm_vec);
    AddInputVec(weight_norm_vec);
    AddInputVec(adam_step_vec);
    AddInputVec(trust_ratio_vec);

    auto hl_neg_step_t = GetOrCreateHbLazyTensor(neg_step_t, c10::kHPU);
    AddInput(hl_neg_step_t.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{
        weight_vec[0],
        adam_norm_vec[0],
        weight_norm_vec[0],
        adam_step_vec[0],
        trust_ratio_vec[0],
        neg_step_t};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        weight_decay,
        static_cast<size_t>(OptimizerFusedLambPhase2Meta::kwdIdx));
    m_meta_data.set(
        use_lamb,
        static_cast<size_t>(OptimizerFusedLambPhase2Meta::kuselambIdx));
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << Node::ToString() << ", beta1="
       << m_meta_data
              .get(static_cast<size_t>(OptimizerFusedLambPhase2Meta::kwdIdx))
              .toDouble()
       << ", beta2="
       << m_meta_data
              .get(static_cast<size_t>(
                  OptimizerFusedLambPhase2Meta::kuselambIdx))
              .toInt();

    return ss.str();
  }

 private:
  void AddInputVec(const std::vector<at::Tensor>& tensor_list) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& t : tensor_list) {
      auto hl_tensor = GetOrCreateHbLazyTensor(t, c10::kHPU);
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
