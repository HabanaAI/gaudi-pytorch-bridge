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

class MaxPool : public ir::Node {
 public:
  enum class MaxPoolParams {
    KERNEL_SIZE_INDEX = 1,
    STRIDE_INDEX,
    PADDING_INDEX,
    DILATION_INDEX,
    CEIL_MODE_INDEX
  };
  MaxPool()
      : Node(c10::Symbol::fromQualString("aten::max_pool2d_with_indices")) {}

  void Init(
      const at::Tensor& input,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool ceil_mode) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHPU);
    hl_input = HbLazyTensorViews::HandleViewsOrUpdate(input, hl_input);
    AddInput(hl_input.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{input};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        kernel_size, static_cast<size_t>(MaxPoolParams::KERNEL_SIZE_INDEX));
    m_meta_data.set(stride, static_cast<size_t>(MaxPoolParams::STRIDE_INDEX));
    m_meta_data.set(padding, static_cast<size_t>(MaxPoolParams::PADDING_INDEX));
    m_meta_data.set(
        dilation, static_cast<size_t>(MaxPoolParams::DILATION_INDEX));
    m_meta_data.set(
        ceil_mode, static_cast<size_t>(MaxPoolParams::CEIL_MODE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", kernel_size="
       << m_meta_data.get(static_cast<size_t>(MaxPoolParams::KERNEL_SIZE_INDEX))
       << ", stride="
       << m_meta_data.get(static_cast<size_t>(MaxPoolParams::STRIDE_INDEX))
       << ", padding="
       << m_meta_data.get(static_cast<size_t>(MaxPoolParams::PADDING_INDEX))
       << ", dilation="
       << m_meta_data.get(static_cast<size_t>(MaxPoolParams::DILATION_INDEX))
       << ", transposed="
       << m_meta_data.get(static_cast<size_t>(MaxPoolParams::CEIL_MODE_INDEX));
    return ss.str();
  }
};

class MaxPoolBackWard : public ir::Node {
 public:
  enum class MaxPoolBwdParams {
    KERNEL_SIZE_INDEX = 2,
    STRIDE_INDEX,
    PADDING_INDEX,
    DILATION_INDEX,
    CEIL_MODE_INDEX
  };
  MaxPoolBackWard()
      : Node(c10::Symbol::fromQualString(
            "aten::max_pool2d_with_indices_backward")) {}
  void Init(
      const at::Tensor& grad_output,
      const at::Tensor& input,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool ceil_mode,
      const at::Tensor& indices) {
    auto hl_grad_output = GetOrCreateHbLazyTensor(grad_output, c10::kHPU);
    hl_grad_output =
        HbLazyTensorViews::HandleViewsOrUpdate(grad_output, hl_grad_output);
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHPU);
    hl_input = HbLazyTensorViews::HandleViewsOrUpdate(input, hl_input);
    auto hl_indices = GetOrCreateHbLazyTensor(indices, c10::kHPU);
    hl_indices = HbLazyTensorViews::HandleViewsOrUpdate(indices, hl_indices);
    AddInput(hl_grad_output.GetIrValue());
    AddInput(hl_input.GetIrValue());
    AddInput(hl_indices.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{grad_output, input, indices};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        kernel_size, static_cast<size_t>(MaxPoolBwdParams::KERNEL_SIZE_INDEX));
    m_meta_data.set(
        stride, static_cast<size_t>(MaxPoolBwdParams::STRIDE_INDEX));
    m_meta_data.set(
        padding, static_cast<size_t>(MaxPoolBwdParams::PADDING_INDEX));
    m_meta_data.set(
        dilation, static_cast<size_t>(MaxPoolBwdParams::DILATION_INDEX));
    m_meta_data.set(
        ceil_mode, static_cast<size_t>(MaxPoolBwdParams::CEIL_MODE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", kernel_size="
       << m_meta_data.get(
              static_cast<size_t>(MaxPoolBwdParams::KERNEL_SIZE_INDEX))
       << ", stride="
       << m_meta_data.get(static_cast<size_t>(MaxPoolBwdParams::STRIDE_INDEX))
       << ", padding="
       << m_meta_data.get(static_cast<size_t>(MaxPoolBwdParams::PADDING_INDEX))
       << ", dilation="
       << m_meta_data.get(static_cast<size_t>(MaxPoolBwdParams::DILATION_INDEX))
       << ", transposed="
       << m_meta_data.get(
              static_cast<size_t>(MaxPoolBwdParams::CEIL_MODE_INDEX));
    return ss.str();
  }
};

class AvgPool : public ir::Node {
 public:
  enum class AvgPoolParams {
    KERNEL_SIZE_INDEX = 1,
    STRIDE_INDEX,
    PADDING_INDEX,
    CEIL_MODE_INDEX,
    COUNT_INCLUDE_PAD,
    DIVISOR_OVERRIDE
  };
  AvgPool() = delete;
  AvgPool(
      const at::Tensor& input,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      bool ceil_mode,
      bool count_include_pad,
      c10::optional<int64_t> divisor_override)
      : Node(c10::Symbol::fromQualString("aten::avg_pool2d")) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHPU);
    hl_input = HbLazyTensorViews::HandleViewsOrUpdate(input, hl_input);
    AddInput(hl_input.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{input};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        kernel_size, static_cast<size_t>(AvgPoolParams::KERNEL_SIZE_INDEX));
    m_meta_data.set(stride, static_cast<size_t>(AvgPoolParams::STRIDE_INDEX));
    m_meta_data.set(padding, static_cast<size_t>(AvgPoolParams::PADDING_INDEX));
    m_meta_data.set(
        ceil_mode, static_cast<size_t>(AvgPoolParams::CEIL_MODE_INDEX));
    m_meta_data.set(
        count_include_pad,
        static_cast<size_t>(AvgPoolParams::COUNT_INCLUDE_PAD));
    m_meta_data.set(
        divisor_override, static_cast<size_t>(AvgPoolParams::DIVISOR_OVERRIDE));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", kernel_size="
       << m_meta_data.get(static_cast<size_t>(AvgPoolParams::KERNEL_SIZE_INDEX))
       << ", stride="
       << m_meta_data.get(static_cast<size_t>(AvgPoolParams::STRIDE_INDEX))
       << ", padding="
       << m_meta_data.get(static_cast<size_t>(AvgPoolParams::PADDING_INDEX))
       << ", transposed="
       << m_meta_data.get(static_cast<size_t>(AvgPoolParams::CEIL_MODE_INDEX))
       << ", include_zero_padding="
       << m_meta_data.get(static_cast<size_t>(AvgPoolParams::COUNT_INCLUDE_PAD))
       << ", divisor_override="
       << m_meta_data.get(static_cast<size_t>(AvgPoolParams::DIVISOR_OVERRIDE));
    return ss.str();
  }
};

class AvgPoolBackWard : public ir::Node {
 public:
  enum class AvgPoolBwdParams {
    KERNEL_SIZE_INDEX = 2,
    STRIDE_INDEX,
    PADDING_INDEX,
    CEIL_MODE_INDEX,
    COUNT_INCLUDE_PAD,
    DIVISOR_OVERRIDE
  };
  AvgPoolBackWard() = delete;
  AvgPoolBackWard(
      const at::Tensor& grad_output,
      const at::Tensor& input,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      bool ceil_mode,
      bool count_include_pad,
      c10::optional<int64_t> divisor_override)
      : Node(c10::Symbol::fromQualString("aten::avg_pool2d_backward")) {
    auto hl_grad_output = GetOrCreateHbLazyTensor(grad_output, c10::kHPU);
    hl_grad_output =
        HbLazyTensorViews::HandleViewsOrUpdate(grad_output, hl_grad_output);
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHPU);
    hl_input = HbLazyTensorViews::HandleViewsOrUpdate(input, hl_input);
    AddInput(hl_grad_output.GetIrValue());
    AddInput(hl_input.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{grad_output, input};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        kernel_size, static_cast<size_t>(AvgPoolBwdParams::KERNEL_SIZE_INDEX));
    m_meta_data.set(
        stride, static_cast<size_t>(AvgPoolBwdParams::STRIDE_INDEX));
    m_meta_data.set(
        padding, static_cast<size_t>(AvgPoolBwdParams::PADDING_INDEX));
    m_meta_data.set(
        ceil_mode, static_cast<size_t>(AvgPoolBwdParams::CEIL_MODE_INDEX));
    m_meta_data.set(
        count_include_pad,
        static_cast<size_t>(AvgPoolBwdParams::COUNT_INCLUDE_PAD));
    m_meta_data.set(
        divisor_override,
        static_cast<size_t>(AvgPoolBwdParams::DIVISOR_OVERRIDE));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", kernel_size="
       << m_meta_data.get(
              static_cast<size_t>(AvgPoolBwdParams::KERNEL_SIZE_INDEX))
       << ", stride="
       << m_meta_data.get(static_cast<size_t>(AvgPoolBwdParams::STRIDE_INDEX))
       << ", padding="
       << m_meta_data.get(static_cast<size_t>(AvgPoolBwdParams::PADDING_INDEX))
       << ", transposed="
       << m_meta_data.get(
              static_cast<size_t>(AvgPoolBwdParams::CEIL_MODE_INDEX))
       << ", include_zero_padding="
       << m_meta_data.get(
              static_cast<size_t>(AvgPoolBwdParams::COUNT_INCLUDE_PAD))
       << ", divisor_override="
       << m_meta_data.get(
              static_cast<size_t>(AvgPoolBwdParams::DIVISOR_OVERRIDE));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
