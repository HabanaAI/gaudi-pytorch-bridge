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
class EmbeddingBagSum : public ir::Node {
 public:
  enum class EmbeddingBagSumParams { KERNEL_MODE_INDEX = 4 };
  EmbeddingBagSum()
      : Node(c10::Symbol::fromQualString("hpu::embedding_bag_sum")) {}
  void Init(
      const at::Tensor& input,
      const at::Tensor& indices,
      const at::Tensor& offsets,
      const at::Tensor& valid_count,
      int64_t kernel_mode) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHPU);
    auto hl_indices = GetOrCreateHbLazyTensor(indices, c10::kHPU);
    auto hl_offsets = GetOrCreateHbLazyTensor(offsets, c10::kHPU);
    auto hl_valid_count = GetOrCreateHbLazyTensor(valid_count, c10::kHPU);

    hl_input = HbLazyTensorViews::HandleViewsOrUpdate(input, hl_input);
    hl_indices = HbLazyTensorViews::HandleViewsOrUpdate(indices, hl_indices);
    hl_offsets = HbLazyTensorViews::HandleViewsOrUpdate(offsets, hl_offsets);
    hl_valid_count =
        HbLazyTensorViews::HandleViewsOrUpdate(valid_count, hl_valid_count);

    AddInput(hl_input.GetIrValue());
    AddInput(hl_indices.GetIrValue());
    AddInput(hl_offsets.GetIrValue());
    AddInput(hl_valid_count.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{input, indices, offsets, valid_count};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        kernel_mode,
        static_cast<size_t>(EmbeddingBagSumParams::KERNEL_MODE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", kernel_mode="
       << m_meta_data.get(
              static_cast<size_t>(EmbeddingBagSumParams::KERNEL_MODE_INDEX));

    return ss.str();
  } // std::string ToString()
}; // class EmbeddingBagSum : public ir::Node

class EmbeddingBagSumBwd : public ir::Node {
 public:
  enum class EmbeddingBagSumBwdParams { KERNEL_MODE_INDEX = 5 };
  EmbeddingBagSumBwd()
      : Node(c10::Symbol::fromQualString("hpu::embedding_bag_sum_bwd_out")) {}

  void Init(
      at::Tensor& out,
      const at::Tensor& input,
      const at::Tensor& indices,
      const at::Tensor& offsets,
      const at::Tensor& valid_count,
      int64_t kernel_mode) {
    auto hl_out = GetOrCreateHbLazyTensor(out, c10::kHPU);
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHPU);
    auto hl_indices = GetOrCreateHbLazyTensor(indices, c10::kHPU);
    auto hl_offsets = GetOrCreateHbLazyTensor(offsets, c10::kHPU);
    auto hl_valid_count = GetOrCreateHbLazyTensor(valid_count, c10::kHPU);

    hl_out = HbLazyTensorViews::HandleViewsOrUpdate(out, hl_out);
    hl_input = HbLazyTensorViews::HandleViewsOrUpdate(input, hl_input);
    hl_indices = HbLazyTensorViews::HandleViewsOrUpdate(indices, hl_indices);
    hl_offsets = HbLazyTensorViews::HandleViewsOrUpdate(offsets, hl_offsets);
    hl_valid_count =
        HbLazyTensorViews::HandleViewsOrUpdate(valid_count, hl_valid_count);

    AddInput(hl_out.GetIrValue());
    AddInput(hl_input.GetIrValue());
    AddInput(hl_indices.GetIrValue());
    AddInput(hl_offsets.GetIrValue());
    AddInput(hl_valid_count.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{
        out, input, indices, offsets, valid_count};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        kernel_mode,
        static_cast<size_t>(EmbeddingBagSumBwdParams::KERNEL_MODE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", kernel_mode="
       << m_meta_data.get(
              static_cast<size_t>(EmbeddingBagSumBwdParams::KERNEL_MODE_INDEX));

    return ss.str();
  } // std::string ToString()
}; // class EmbeddingBagSumBwd : public ir::Node
} // namespace ir
}; // namespace habana_lazy
