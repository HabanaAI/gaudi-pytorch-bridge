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

class MaxPool : public ir::Node {
 public:
 enum class MaxPoolParams { KERNEL_SIZE_INDEX=1,
                            STRIDE_INDEX,
                            PADDING_INDEX,
                            DILATION_INDEX,
                            CEIL_MODE_INDEX};
  MaxPool() = delete;
  MaxPool(
      const at::Tensor& input,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool ceil_mode)
      : Node(c10::Symbol::fromQualString("aten::max_pool2d_with_indices")) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHABANA);
    AddInput(hl_input.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{input};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(kernel_size, static_cast<size_t>(MaxPoolParams::KERNEL_SIZE_INDEX));
    m_meta_data.set(stride, static_cast<size_t>(MaxPoolParams::STRIDE_INDEX));
    m_meta_data.set(padding, static_cast<size_t>(MaxPoolParams::PADDING_INDEX));
    m_meta_data.set(dilation, static_cast<size_t>(MaxPoolParams::DILATION_INDEX));
    m_meta_data.set(ceil_mode, static_cast<size_t>(MaxPoolParams::CEIL_MODE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", kernel_size=" << m_meta_data.get(static_cast<size_t>(MaxPoolParams::KERNEL_SIZE_INDEX))
       << ", stride=" << m_meta_data.get(static_cast<size_t>(MaxPoolParams::STRIDE_INDEX))
       << ", padding=" << m_meta_data.get(static_cast<size_t>(MaxPoolParams::PADDING_INDEX))
       << ", dilation=" << m_meta_data.get(static_cast<size_t>(MaxPoolParams::DILATION_INDEX))
       << ", transposed=" << m_meta_data.get(static_cast<size_t>(MaxPoolParams::CEIL_MODE_INDEX));
    return ss.str();
  }
};

class MaxPoolBackWard : public ir::Node {
 public:
 enum class MaxPoolBwdParams { KERNEL_SIZE_INDEX=2,
                               STRIDE_INDEX,
                               PADDING_INDEX,
                               DILATION_INDEX,
                               CEIL_MODE_INDEX};
  MaxPoolBackWard() = delete;
  MaxPoolBackWard(
      const at::Tensor& grad_output,
      const at::Tensor& input,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool ceil_mode,
      const at::Tensor& indices)
      : Node(c10::Symbol::fromQualString("aten::max_pool2d_with_indices_backward")) {
    auto hl_grad_output = GetOrCreateHbLazyTensor(grad_output, c10::kHABANA);
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHABANA);
    auto hl_indices = GetOrCreateHbLazyTensor(indices, c10::kHABANA);
    AddInput(hl_grad_output.GetIrValue());
    AddInput(hl_input.GetIrValue());
    AddInput(hl_indices.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{grad_output, input, indices};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(kernel_size, static_cast<size_t>(MaxPoolBwdParams::KERNEL_SIZE_INDEX));
    m_meta_data.set(stride, static_cast<size_t>(MaxPoolBwdParams::STRIDE_INDEX));
    m_meta_data.set(padding, static_cast<size_t>(MaxPoolBwdParams::PADDING_INDEX));
    m_meta_data.set(dilation, static_cast<size_t>(MaxPoolBwdParams::DILATION_INDEX));
    m_meta_data.set(ceil_mode, static_cast<size_t>(MaxPoolBwdParams::CEIL_MODE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", kernel_size=" << m_meta_data.get(static_cast<size_t>(MaxPoolBwdParams::KERNEL_SIZE_INDEX))
       << ", stride=" << m_meta_data.get(static_cast<size_t>(MaxPoolBwdParams::STRIDE_INDEX))
       << ", padding=" << m_meta_data.get(static_cast<size_t>(MaxPoolBwdParams::PADDING_INDEX))
       << ", dilation=" << m_meta_data.get(static_cast<size_t>(MaxPoolBwdParams::DILATION_INDEX))
       << ", transposed=" << m_meta_data.get(static_cast<size_t>(MaxPoolBwdParams::CEIL_MODE_INDEX));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
