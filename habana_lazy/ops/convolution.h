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
  Convolution(
      const at::Tensor& input,
      const at::Tensor& weight,
      const at::Tensor& bias,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool transposed,
      at::IntArrayRef output_padding,
      int64_t groups)
      : Node(c10::Symbol::fromQualString("aten::convolution_overrideable")) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHABANA);
    auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);

    AddInput(hl_input.GetIrValue());
    AddInput(hl_weight.GetIrValue());

    if (bias.defined()) {
      auto hl_bias = GetOrCreateHbLazyTensor(bias, c10::kHABANA);
      AddInput(hl_bias.GetIrValue());
    } else {
      m_meta_data.set(
          torch::jit::IValue(), static_cast<size_t>(ConvParams::BIAS_INDEX));
    }

    m_meta_data.set(stride, static_cast<size_t>(ConvParams::STRIDE_INDEX));
    m_meta_data.set(padding, static_cast<size_t>(ConvParams::PADDING_INDEX));
    m_meta_data.set(dilation, static_cast<size_t>(ConvParams::DILATION_INDEX));
    m_meta_data.set(
        transposed, static_cast<size_t>(ConvParams::TRANSPOSED_INDEX));
    m_meta_data.set(
        output_padding, static_cast<size_t>(ConvParams::OUTPUT_PADDING_INDEX));
    m_meta_data.set(groups, static_cast<size_t>(ConvParams::GROUPS_INDEX));
  }

  Convolution(
      const at::Tensor& grad_output,
      const at::Tensor& input,
      const at::Tensor& weight,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool transposed,
      at::IntArrayRef output_padding,
      int64_t groups,
      std::vector<bool> output_mask)
      : Node(c10::Symbol::fromQualString(
            "aten::convolution_backward_overrideable")) {
    auto hl_grad_output = GetOrCreateHbLazyTensor(grad_output, c10::kHABANA);
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHABANA);
    auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);

    AddInput(hl_grad_output.GetIrValue());
    AddInput(hl_input.GetIrValue());
    AddInput(hl_weight.GetIrValue());

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
