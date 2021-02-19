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

class LayerNormForward : public ir::Node {
 public:
  enum class LayerNormForwardMeta {
    WEIGHT_INDEX = 1,
    BIAS_INDEX,
    M_INDEX,
    N_INDEX,
    EPS_INDEX
  };
  LayerNormForward() = delete;
  LayerNormForward(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias,
      int64_t m,
      int64_t n,
      double eps)
      : Node(c10::Symbol::fromQualString("aten::native_layer_norm")) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHABANA);
    AddInput(hl_input.GetIrValue());
    std::vector<at::Tensor> input_pt_vec{input};
    if (weight.defined()) {
      auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
      input_pt_vec.emplace_back(weight);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(LayerNormForwardMeta::WEIGHT_INDEX));
    }
    if (bias.defined()) {
      auto hl_bias = GetOrCreateHbLazyTensor(bias, c10::kHABANA);
      AddInput(hl_bias.GetIrValue());
      input_pt_vec.emplace_back(bias);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(LayerNormForwardMeta::BIAS_INDEX));
    }
    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(m, static_cast<size_t>(LayerNormForwardMeta::M_INDEX));
    m_meta_data.set(n, static_cast<size_t>(LayerNormForwardMeta::N_INDEX));
    m_meta_data.set(eps, static_cast<size_t>(LayerNormForwardMeta::EPS_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", M="
       << m_meta_data.get(static_cast<size_t>(LayerNormForwardMeta::M_INDEX))
       << ", N="
       << m_meta_data.get(static_cast<size_t>(LayerNormForwardMeta::N_INDEX))
       << ", EPS="
       << m_meta_data.get(static_cast<size_t>(LayerNormForwardMeta::EPS_INDEX));
    return ss.str();
  }
};

class LayerNormBackward : public ir::Node {
 public:
  enum class LayerNormBackwardMeta {
    GAMMA_INDEX = 4,
    M_INDEX,
    N_INDEX,
    MASK_INDEX
  };
  LayerNormBackward() = delete;
  LayerNormBackward(
      const Tensor& dY,
      const Tensor& X,
      const Tensor& mean,
      const Tensor& rstd,
      const Tensor& gamma,
      int64_t M,
      int64_t N,
      std::array<bool, 3> grad_input_mask)
      : Node(c10::Symbol::fromQualString("aten::native_layer_norm_backward")) {
    auto hl_dY = GetOrCreateHbLazyTensor(dY, c10::kHABANA);
    AddInput(hl_dY.GetIrValue());
    auto hl_X = GetOrCreateHbLazyTensor(X, c10::kHABANA);
    AddInput(hl_X.GetIrValue());
    auto hl_mean = GetOrCreateHbLazyTensor(mean, c10::kHABANA);
    AddInput(hl_mean.GetIrValue());
    auto hl_rstd = GetOrCreateHbLazyTensor(rstd, c10::kHABANA);
    AddInput(hl_rstd.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{dY, X, mean, rstd};

    if (gamma.defined()) {
      auto hl_gamma = GetOrCreateHbLazyTensor(gamma, c10::kHABANA);
      AddInput(hl_gamma.GetIrValue());
      input_pt_vec.emplace_back(gamma);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(LayerNormBackwardMeta::GAMMA_INDEX));
    }

    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(M, static_cast<size_t>(LayerNormBackwardMeta::M_INDEX));
    m_meta_data.set(N, static_cast<size_t>(LayerNormBackwardMeta::N_INDEX));
    c10::List<bool> boolList{
        grad_input_mask[0], grad_input_mask[1], grad_input_mask[2]};
    m_meta_data.set(
        boolList, static_cast<size_t>(LayerNormBackwardMeta::MASK_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", M="
       << m_meta_data.get(static_cast<size_t>(LayerNormBackwardMeta::M_INDEX))
       << ", N="
       << m_meta_data.get(static_cast<size_t>(LayerNormBackwardMeta::N_INDEX))
       << ", Output Mask="
       << m_meta_data.get(
              static_cast<size_t>(LayerNormBackwardMeta::MASK_INDEX));
    return ss.str();
  }
};

class BatchNormForward : public ir::Node {
 public:
  enum class BatchNormForwardMeta {
    WEIGHT_INDEX = 1,
    BIAS_INDEX,
    RUNNING_MEAN_INDEX,
    RUNNING_VAR_INDEX,
    TRAINING_INDEX = 5,
    MOMENTUM_INDEX,
    EPS_INDEX
  };
  BatchNormForward() = delete;
  BatchNormForward(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias,
      const Tensor& running_mean,
      const Tensor& running_var,
      bool training,
      double momentum,
      double eps)
      : Node(c10::Symbol::fromQualString("hpu::native_batch_norm_rmv")) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHABANA);
    AddInput(hl_input.GetIrValue());
    std::vector<at::Tensor> input_pt_vec{input};
    if (weight.defined()) {
      auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
      input_pt_vec.emplace_back(weight);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormForwardMeta::WEIGHT_INDEX));
    }
    if (bias.defined()) {
      auto hl_bias = GetOrCreateHbLazyTensor(bias, c10::kHABANA);
      AddInput(hl_bias.GetIrValue());
      input_pt_vec.emplace_back(bias);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormForwardMeta::BIAS_INDEX));
    }
    if (running_mean.defined()) {
      auto hl_running_mean =
          GetOrCreateHbLazyTensor(running_mean, c10::kHABANA);
      AddInput(hl_running_mean.GetIrValue());
      input_pt_vec.emplace_back(running_mean);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormForwardMeta::RUNNING_MEAN_INDEX));
    }
    if (running_var.defined()) {
      auto hl_running_var = GetOrCreateHbLazyTensor(running_var, c10::kHABANA);
      AddInput(hl_running_var.GetIrValue());
      input_pt_vec.emplace_back(running_var);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormForwardMeta::RUNNING_VAR_INDEX));
    }
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        training, static_cast<size_t>(BatchNormForwardMeta::TRAINING_INDEX));
    m_meta_data.set(
        momentum, static_cast<size_t>(BatchNormForwardMeta::MOMENTUM_INDEX));
    m_meta_data.set(eps, static_cast<size_t>(BatchNormForwardMeta::EPS_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", Training = "
       << m_meta_data.get(
              static_cast<size_t>(BatchNormForwardMeta::TRAINING_INDEX))
       << ", Momentum = "
       << m_meta_data.get(
              static_cast<size_t>(BatchNormForwardMeta::MOMENTUM_INDEX))
       << ", EPS="
       << m_meta_data.get(static_cast<size_t>(BatchNormForwardMeta::EPS_INDEX));
    return ss.str();
  }
};

class BatchNormInf : public ir::Node {
 public:
  enum class BatchNormInfMeta {
    BIAS_INDEX = 1,
    WEIGHT_INDEX = 2,
    RUNNING_MEAN_INDEX = 3,
    RUNNING_VAR_INDEX = 4,
    TRAINING_INDEX = 5,
    MOMENTUM_INDEX = 6,
    EPS_INDEX = 7
  };
  BatchNormInf() = delete;
  BatchNormInf(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias,
      const Tensor& running_mean,
      const Tensor& running_var,
      bool training,
      double momentum,
      double eps)
      : Node(c10::Symbol::fromQualString("hpu::native_batch_norm_inf")) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHABANA);
    AddInput(hl_input.GetIrValue());
    std::vector<at::Tensor> input_pt_vec{input};
    if (bias.defined()) {
      auto hl_bias = GetOrCreateHbLazyTensor(bias, c10::kHABANA);
      AddInput(hl_bias.GetIrValue());
      input_pt_vec.emplace_back(bias);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormInfMeta::BIAS_INDEX));
    }
    if (weight.defined()) {
      auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
      input_pt_vec.emplace_back(weight);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormInfMeta::WEIGHT_INDEX));
    }
    if (running_mean.defined()) {
      auto hl_running_mean =
          GetOrCreateHbLazyTensor(running_mean, c10::kHABANA);
      AddInput(hl_running_mean.GetIrValue());
      input_pt_vec.emplace_back(running_mean);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormInfMeta::RUNNING_MEAN_INDEX));
    }
    if (running_var.defined()) {
      auto hl_running_var = GetOrCreateHbLazyTensor(running_var, c10::kHABANA);
      AddInput(hl_running_var.GetIrValue());
      input_pt_vec.emplace_back(running_var);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormInfMeta::RUNNING_VAR_INDEX));
    }
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        training, static_cast<size_t>(BatchNormInfMeta::TRAINING_INDEX));
    m_meta_data.set(
        momentum, static_cast<size_t>(BatchNormInfMeta::MOMENTUM_INDEX));
    m_meta_data.set(eps, static_cast<size_t>(BatchNormInfMeta::EPS_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", Training = "
       << m_meta_data.get(static_cast<size_t>(BatchNormInfMeta::TRAINING_INDEX))
       << ", Momentum = "
       << m_meta_data.get(static_cast<size_t>(BatchNormInfMeta::MOMENTUM_INDEX))
       << ", EPS="
       << m_meta_data.get(static_cast<size_t>(BatchNormInfMeta::EPS_INDEX));
    return ss.str();
  }
};

class BatchNormBackward : public ir::Node {
 public:
  enum class BatchNormBackwardMeta {
    WEIGHT_INDEX = 2,
    RUNNING_MEAN_INDEX,
    RUNNING_VAR_INDEX,
    SAVE_MEAN_INDEX,
    SAVE_INVSTD_INDEX,
    TRAIN_INDEX,
    EPS_INDEX,
    OUT_MASK_INDEX
  };
  BatchNormBackward() = delete;
  BatchNormBackward(
      const Tensor& grad_out,
      const Tensor& input,
      const Tensor& weight,
      UNUSED const Tensor& running_mean,
      UNUSED const Tensor& running_var,
      const Tensor& save_mean,
      const Tensor& save_invstd,
      bool train,
      double eps,
      UNUSED std::array<bool, 3> output_mask)
      : Node(c10::Symbol::fromQualString("aten::native_batch_norm_backward")) {
    auto hl_grad_out = GetOrCreateHbLazyTensor(grad_out, c10::kHABANA);
    AddInput(hl_grad_out.GetIrValue());
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHABANA);
    AddInput(hl_input.GetIrValue());
    std::vector<at::Tensor> input_pt_vec{grad_out, input};
    if (weight.defined()) {
      auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
      input_pt_vec.emplace_back(weight);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormBackwardMeta::WEIGHT_INDEX));
    }

    if (running_mean.defined()) {
      auto hl_running_mean =
          GetOrCreateHbLazyTensor(running_mean, c10::kHABANA);
      AddInput(hl_running_mean.GetIrValue());
      input_pt_vec.emplace_back(running_mean);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormBackwardMeta::RUNNING_MEAN_INDEX));
    }
    if (running_var.defined()) {
      auto hl_running_var = GetOrCreateHbLazyTensor(running_var, c10::kHABANA);
      AddInput(hl_running_var.GetIrValue());
      input_pt_vec.emplace_back(running_var);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormBackwardMeta::RUNNING_VAR_INDEX));
    }

    if (save_mean.defined()) {
      auto hl_save_mean = GetOrCreateHbLazyTensor(save_mean, c10::kHABANA);
      AddInput(hl_save_mean.GetIrValue());
      input_pt_vec.emplace_back(save_mean);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormBackwardMeta::SAVE_MEAN_INDEX));
    }
    if (save_invstd.defined()) {
      auto hl_save_invstd = GetOrCreateHbLazyTensor(save_invstd, c10::kHABANA);
      AddInput(hl_save_invstd.GetIrValue());
      input_pt_vec.emplace_back(save_invstd);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BatchNormBackwardMeta::SAVE_INVSTD_INDEX));
    }

    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        train, static_cast<size_t>(BatchNormBackwardMeta::TRAIN_INDEX));
    m_meta_data.set(eps, static_cast<size_t>(BatchNormBackwardMeta::EPS_INDEX));
    c10::List<bool> boolList{output_mask[0], output_mask[1], output_mask[2]};
    m_meta_data.set(
        boolList, static_cast<size_t>(BatchNormBackwardMeta::OUT_MASK_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", Train = "
       << m_meta_data.get(
              static_cast<size_t>(BatchNormBackwardMeta::TRAIN_INDEX))
       << ", EPS = "
       << m_meta_data.get(static_cast<size_t>(BatchNormBackwardMeta::EPS_INDEX))
       << ", OUT_MASK = "
       << m_meta_data.get(
              static_cast<size_t>(BatchNormBackwardMeta::OUT_MASK_INDEX));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
