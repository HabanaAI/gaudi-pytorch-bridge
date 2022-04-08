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

class LayerNormForward : public ir::Node {
 public:
  enum class LayerNormForwardMeta {
    NORMALIZED_INDEX = 1,
    WEIGHT_INDEX,
    BIAS_INDEX,
    EPS_INDEX
  };
  LayerNormForward() = delete;
  LayerNormForward(
      const at::Tensor& input,
      at::IntArrayRef normalized_shape,
      const c10::optional<at::Tensor>& weight_opt,
      const c10::optional<at::Tensor>& bias_opt,
      double eps)
      : Node(c10::Symbol::fromQualString("aten::native_layer_norm")) {
    auto weight = weight_opt.value();
    auto bias = bias_opt.value();
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHPU);
    hl_input = HbLazyTensorViews::HandleViewsOrUpdate(input, hl_input);
    AddInput(hl_input.GetIrValue());
    std::vector<at::Tensor> input_pt_vec{input};
    auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHPU);
    hl_weight = HbLazyTensorViews::HandleViewsOrUpdate(weight, hl_weight);
    AddInput(hl_weight.GetIrValue());
    input_pt_vec.emplace_back(weight);

    auto hl_bias = GetOrCreateHbLazyTensor(bias, c10::kHPU);
    hl_bias = HbLazyTensorViews::HandleViewsOrUpdate(bias, hl_bias);
    AddInput(hl_bias.GetIrValue());
    input_pt_vec.emplace_back(bias);

    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(
        normalized_shape,
        static_cast<size_t>(LayerNormForwardMeta::NORMALIZED_INDEX));
    m_meta_data.set(eps, static_cast<size_t>(LayerNormForwardMeta::EPS_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", normalized_shape="
       << m_meta_data.get(
              static_cast<size_t>(LayerNormForwardMeta::NORMALIZED_INDEX))
       << ", EPS="
       << m_meta_data.get(static_cast<size_t>(LayerNormForwardMeta::EPS_INDEX));
    return ss.str();
  }
};

class LayerNormBackward : public ir::Node {
 public:
  enum class LayerNormBackwardMeta {
    NORMALIZED_INDEX = 2,
    MEAN_INDEX,
    RSTD_INDEX,
    WEIGHT_INDEX,
    BIAS_INDEX,
    MASK_INDEX
  };
  LayerNormBackward() = delete;
  LayerNormBackward(
      const at::Tensor& dY,
      const at::Tensor& X,
      at::IntArrayRef normalized_shape,
      const at::Tensor& mean,
      const at::Tensor& rstd,
      const c10::optional<at::Tensor>& weight_opt,
      const c10::optional<at::Tensor>& bias_opt,
      std::array<bool, 3> grad_input_mask)
      : Node(c10::Symbol::fromQualString("aten::native_layer_norm_backward")) {
    auto hl_dY = GetOrCreateHbLazyTensor(dY, c10::kHPU);
    hl_dY = HbLazyTensorViews::HandleViewsOrUpdate(dY, hl_dY);
    AddInput(hl_dY.GetIrValue());
    auto hl_X = GetOrCreateHbLazyTensor(X, c10::kHPU);
    hl_X = HbLazyTensorViews::HandleViewsOrUpdate(X, hl_X);
    AddInput(hl_X.GetIrValue());
    auto hl_mean = GetOrCreateHbLazyTensor(mean, c10::kHPU);
    hl_mean = HbLazyTensorViews::HandleViewsOrUpdate(mean, hl_mean);
    AddInput(hl_mean.GetIrValue());
    auto hl_rstd = GetOrCreateHbLazyTensor(rstd, c10::kHPU);
    hl_rstd = HbLazyTensorViews::HandleViewsOrUpdate(rstd, hl_rstd);
    AddInput(hl_rstd.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{dY, X, mean, rstd};
    auto gamma = weight_opt.value();

    auto hl_gamma = GetOrCreateHbLazyTensor(gamma, c10::kHPU);
    hl_gamma = HbLazyTensorViews::HandleViewsOrUpdate(gamma, hl_gamma);
    AddInput(hl_gamma.GetIrValue());
    input_pt_vec.emplace_back(gamma);
    auto bias = bias_opt.value();

    auto hl_bias = GetOrCreateHbLazyTensor(bias, c10::kHPU);
    hl_bias = HbLazyTensorViews::HandleViewsOrUpdate(bias, hl_bias);
    AddInput(hl_bias.GetIrValue());
    input_pt_vec.emplace_back(bias);
    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(
        normalized_shape,
        static_cast<size_t>(LayerNormBackwardMeta::NORMALIZED_INDEX));
    c10::List<bool> boolList{
        grad_input_mask[0], grad_input_mask[1], grad_input_mask[2]};
    m_meta_data.set(
        boolList, static_cast<size_t>(LayerNormBackwardMeta::MASK_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", normalized_shape="
       << m_meta_data.get(
              static_cast<size_t>(LayerNormBackwardMeta::NORMALIZED_INDEX))

       << ", Output Mask="
       << m_meta_data.get(
              static_cast<size_t>(LayerNormBackwardMeta::MASK_INDEX));
    return ss.str();
  }
};

class FusedNorm : public ir::Node {
 public:
  enum class FusedNormMeta {
    NORM_TYPE_INDEX = 2,
  };
  FusedNorm() = delete;
  FusedNorm(
      std::vector<at::Tensor>& grad,
      const at::Tensor& max_norm,
      float norm_type)
      : Node(c10::Symbol::fromQualString("hpu::fused_norm_lazy")) {
    AddInputVec(grad);

    auto hl_max_norm = GetOrCreateHbLazyTensor(max_norm, c10::kHPU);
    hl_max_norm = HbLazyTensorViews::HandleViewsOrUpdate(max_norm, hl_max_norm);
    AddInput(hl_max_norm.GetIrValue());

    m_meta_data.set(
        norm_type, static_cast<size_t>(FusedNormMeta::NORM_TYPE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", norm_type = "
       << m_meta_data.get(static_cast<size_t>(FusedNormMeta::NORM_TYPE_INDEX));
    return ss.str();
  }

 private:
  void AddInputVec(std::vector<at::Tensor>& tensor_list) {
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
