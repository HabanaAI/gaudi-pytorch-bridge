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
    if (weight.defined()) {
      auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(LayerNormForwardMeta::WEIGHT_INDEX));
    }
    if (bias.defined()) {
      auto hl_bias = GetOrCreateHbLazyTensor(bias, c10::kHABANA);
      AddInput(hl_bias.GetIrValue());
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(LayerNormForwardMeta::BIAS_INDEX));
    }
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

    if (gamma.defined()) {
      auto hl_gamma = GetOrCreateHbLazyTensor(gamma, c10::kHABANA);
      AddInput(hl_gamma.GetIrValue());
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(LayerNormBackwardMeta::GAMMA_INDEX));
    }
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

}; // namespace ir
}; // namespace habana_lazy