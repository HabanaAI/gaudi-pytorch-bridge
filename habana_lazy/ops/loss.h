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

class NllLoss_forward : public ir::Node {
 public:
  enum class NllLossParams { WEIGHT_INDEX = 2, REDUCTION_INDEX, IGNORE_INDEX };
  NllLoss_forward() = delete;
  NllLoss_forward(
      const at::Tensor& self,
      const at::Tensor& target,
      const at::Tensor& weight,
      int64_t reduction,
      int64_t ignore_index)
      : Node(c10::Symbol::fromQualString("aten::nll_loss_forward")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_self.GetIrValue());

    auto hl_target = GetOrCreateHbLazyTensor(target, c10::kHABANA);
    AddInput(hl_target.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self, target};

    if (weight.defined()) {
      auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
      input_pt_vec.emplace_back(weight);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(NllLossParams::WEIGHT_INDEX));
    }

    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(
        reduction, static_cast<size_t>(NllLossParams::REDUCTION_INDEX));
    m_meta_data.set(
        ignore_index, static_cast<size_t>(NllLossParams::IGNORE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", reduction="
       << m_meta_data.get(static_cast<size_t>(NllLossParams::REDUCTION_INDEX))
       << ", ignore_index="
       << m_meta_data.get(static_cast<size_t>(NllLossParams::IGNORE_INDEX));
    return ss.str();
  }
};

class NllLoss_backward : public ir::Node {
 public:
  enum class NllLossParams {
    WEIGHT_INDEX = 3,
    REDUCTION_INDEX,
    IGNORE_INDEX,
    TOTAL_WEIGHT_INDEX
  };
  NllLoss_backward() = delete;
  NllLoss_backward(
      const at::Tensor& grad_output,
      const at::Tensor& self,
      const at::Tensor& target,
      const at::Tensor& weight,
      int64_t reduction,
      int64_t ignore_index,
      const at::Tensor& total_weight)
      : Node(c10::Symbol::fromQualString("aten::nll_loss_backward")) {
    auto hl_grad_output = GetOrCreateHbLazyTensor(grad_output, c10::kHABANA);
    AddInput(hl_grad_output.GetIrValue());
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_self.GetIrValue());
    auto hl_target = GetOrCreateHbLazyTensor(target, c10::kHABANA);
    AddInput(hl_target.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{grad_output, self, target};

    if (weight.defined()) {
      auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
      input_pt_vec.emplace_back(weight);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(NllLossParams::WEIGHT_INDEX));
    }

    if (total_weight.defined()) {
      auto hl_total_weight =
          GetOrCreateHbLazyTensor(total_weight, c10::kHABANA);
      AddInput(hl_total_weight.GetIrValue());
      input_pt_vec.emplace_back(total_weight);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(NllLossParams::TOTAL_WEIGHT_INDEX));
    }

    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(
        reduction, static_cast<size_t>(NllLossParams::REDUCTION_INDEX));
    m_meta_data.set(
        ignore_index, static_cast<size_t>(NllLossParams::IGNORE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", reduction="
       << m_meta_data.get(static_cast<size_t>(NllLossParams::REDUCTION_INDEX))
       << ", ignore_index="
       << m_meta_data.get(static_cast<size_t>(NllLossParams::IGNORE_INDEX));
    return ss.str();
  }
};

class BceLoss_forward : public ir::Node {
 public:
  enum class BceLossParams { WEIGHT_INDEX = 2, REDUCTION_INDEX };
  BceLoss_forward() = delete;
  BceLoss_forward(
      const at::Tensor& self,
      const at::Tensor& target,
      const at::Tensor& weight,
      int64_t reduction)
      : Node(c10::Symbol::fromQualString("aten::binary_cross_entropy")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_self.GetIrValue());

    auto hl_target = GetOrCreateHbLazyTensor(target, c10::kHABANA);
    AddInput(hl_target.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self, target};

    if (weight.defined()) {
      auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
      input_pt_vec.emplace_back(weight);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BceLossParams::WEIGHT_INDEX));
    }

    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(
        reduction, static_cast<size_t>(BceLossParams::REDUCTION_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", reduction="
       << m_meta_data.get(static_cast<size_t>(BceLossParams::REDUCTION_INDEX));
    return ss.str();
  }
};

class BceLoss_backward : public ir::Node {
 public:
  enum class BceLossParams { WEIGHT_INDEX = 3, REDUCTION_INDEX };
  BceLoss_backward() = delete;
  BceLoss_backward(
      const at::Tensor& grad_output,
      const at::Tensor& self,
      const at::Tensor& target,
      const at::Tensor& weight,
      int64_t reduction)
      : Node(c10::Symbol::fromQualString(
            "aten::binary_cross_entropy_backward")) {
    auto hl_grad_output = GetOrCreateHbLazyTensor(grad_output, c10::kHABANA);
    AddInput(hl_grad_output.GetIrValue());

    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_self.GetIrValue());

    auto hl_target = GetOrCreateHbLazyTensor(target, c10::kHABANA);
    AddInput(hl_target.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{grad_output, self, target};

    if (weight.defined()) {
      auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
      input_pt_vec.emplace_back(weight);
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BceLossParams::WEIGHT_INDEX));
    }

    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(
        reduction, static_cast<size_t>(BceLossParams::REDUCTION_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", reduction="
       << m_meta_data.get(static_cast<size_t>(BceLossParams::REDUCTION_INDEX));
    return ss.str();
  }
};

class BceLogitsLoss_forward : public ir::Node {
 public:
  enum class BceLossParams {
    WEIGHT_INDEX = 2,
    POS_WEIGHT_INDEX,
    REDUCTION_INDEX
  };
  BceLogitsLoss_forward() = delete;
  BceLogitsLoss_forward(
      const at::Tensor& self,
      const at::Tensor& target,
      const c10::optional<at::Tensor>& weight,
      const c10::optional<at::Tensor>& pos_weight,
      int64_t reduction)
      : Node(c10::Symbol::fromQualString(
            "aten::binary_cross_entropy_with_logits")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_self.GetIrValue());

    auto hl_target = GetOrCreateHbLazyTensor(target, c10::kHABANA);
    AddInput(hl_target.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self, target};

    if (weight.has_value()) {
      auto hl_weight = GetOrCreateHbLazyTensor(weight.value(), c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
      input_pt_vec.emplace_back(weight.value());
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BceLossParams::WEIGHT_INDEX));
    }
    if (pos_weight.has_value()) {
      auto hl_weight =
          GetOrCreateHbLazyTensor(pos_weight.value(), c10::kHABANA);
      AddInput(hl_weight.GetIrValue());
      input_pt_vec.emplace_back(pos_weight.value());
    } else {
      m_meta_data.set(
          torch::jit::IValue(),
          static_cast<size_t>(BceLossParams::POS_WEIGHT_INDEX));
    }

    AddInputPtTensors(input_pt_vec);
    m_meta_data.set(
        reduction, static_cast<size_t>(BceLossParams::REDUCTION_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", reduction="
       << m_meta_data.get(static_cast<size_t>(BceLossParams::REDUCTION_INDEX));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
