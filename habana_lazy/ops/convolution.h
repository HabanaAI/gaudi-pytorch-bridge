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

class Convolution : public ir::Node {
 public:
  enum class ConvParams {
    BIAS_INDEX = 2,
    STRIDE_INDEX,
    PADDING_INDEX,
    DILATION_INDEX,
    TRANSPOSED_INDEX,
    OUTPUT_PADDING_INDEX,
    GROUPS_INDEX,
    OUTPUT_MASK_INDEX
  };
  Convolution() = delete;
  Convolution(const std::string& qual_str)
      : Node(c10::Symbol::fromQualString(qual_str)) {}

  void Init(
      const at::Tensor& input,
      const at::Tensor& weight,
      const at::Tensor& bias,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool transposed,
      at::IntArrayRef output_padding,
      int64_t groups) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHPU);
    auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHPU);

    hl_input = HbLazyTensorViews::HandleViewsOrUpdate(input, hl_input);
    hl_weight = HbLazyTensorViews::HandleViewsOrUpdate(weight, hl_weight);

    AddInput(hl_input.GetIrValue());
    AddInput(hl_weight.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{input, weight};

    if (bias.defined()) {
      auto hl_bias = GetOrCreateHbLazyTensor(bias, c10::kHPU);
      hl_bias = HbLazyTensorViews::HandleViewsOrUpdate(bias, hl_bias);
      AddInput(hl_bias.GetIrValue());
      input_pt_vec.emplace_back(bias);
    } else {
      m_meta_data.set(
          torch::jit::IValue(), static_cast<size_t>(ConvParams::BIAS_INDEX));
    }

    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(stride, static_cast<size_t>(ConvParams::STRIDE_INDEX));
    m_meta_data.set(padding, static_cast<size_t>(ConvParams::PADDING_INDEX));
    m_meta_data.set(dilation, static_cast<size_t>(ConvParams::DILATION_INDEX));
    m_meta_data.set(
        transposed, static_cast<size_t>(ConvParams::TRANSPOSED_INDEX));
    m_meta_data.set(
        output_padding, static_cast<size_t>(ConvParams::OUTPUT_PADDING_INDEX));
    m_meta_data.set(groups, static_cast<size_t>(ConvParams::GROUPS_INDEX));
  }

  void Init(
      const at::Tensor& grad_output,
      const at::Tensor& input,
      const at::Tensor& weight,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool transposed,
      at::IntArrayRef output_padding,
      int64_t groups,
      std::vector<bool> output_mask) {
    auto hl_grad_output = GetOrCreateHbLazyTensor(grad_output, c10::kHPU);
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHPU);
    auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHPU);

    hl_grad_output =
        HbLazyTensorViews::HandleViewsOrUpdate(grad_output, hl_grad_output);
    hl_input = HbLazyTensorViews::HandleViewsOrUpdate(input, hl_input);
    hl_weight = HbLazyTensorViews::HandleViewsOrUpdate(weight, hl_weight);

    AddInput(hl_grad_output.GetIrValue());
    AddInput(hl_input.GetIrValue());
    AddInput(hl_weight.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{grad_output, input, weight};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(stride, static_cast<size_t>(ConvParams::STRIDE_INDEX));
    m_meta_data.set(padding, static_cast<size_t>(ConvParams::PADDING_INDEX));
    m_meta_data.set(dilation, static_cast<size_t>(ConvParams::DILATION_INDEX));
    m_meta_data.set(
        transposed, static_cast<size_t>(ConvParams::TRANSPOSED_INDEX));
    m_meta_data.set(
        output_padding, static_cast<size_t>(ConvParams::OUTPUT_PADDING_INDEX));
    m_meta_data.set(groups, static_cast<size_t>(ConvParams::GROUPS_INDEX));
    m_meta_data.set(
        output_mask, static_cast<size_t>(ConvParams::OUTPUT_MASK_INDEX));
  }
  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", stride="
       << m_meta_data.get(static_cast<size_t>(ConvParams::STRIDE_INDEX))
       << ", padding="
       << m_meta_data.get(static_cast<size_t>(ConvParams::PADDING_INDEX))
       << ", dilation="
       << m_meta_data.get(static_cast<size_t>(ConvParams::DILATION_INDEX))
       << ", transposed="
       << m_meta_data.get(static_cast<size_t>(ConvParams::TRANSPOSED_INDEX))
       << ", output_padding="
       << m_meta_data.get(static_cast<size_t>(ConvParams::OUTPUT_PADDING_INDEX))
       << ", groups="
       << m_meta_data.get(static_cast<size_t>(ConvParams::GROUPS_INDEX));
    if (m_meta_data.count(static_cast<size_t>(ConvParams::OUTPUT_MASK_INDEX))) {
      ss << ", output_mask="
         << m_meta_data.get(static_cast<size_t>(ConvParams::OUTPUT_MASK_INDEX));
    }
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
