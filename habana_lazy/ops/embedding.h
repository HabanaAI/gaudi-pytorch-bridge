/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#pragma once
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana_lazy {
namespace ir {

class Embedding_backward : public ir::Node {
 public:
  enum class EmbeddingBwdParams {
    NUM_WEIGHTS_INDEX = 2,
    PADDING_INDEX,
    SCALE_GRADE_BY_FREQ_INDEX
  };
  Embedding_backward()
      : Node(c10::Symbol::fromQualString("aten::embedding_dense_backward")) {}
  void Init(
      const at::Tensor& grad,
      const at::Tensor& indices,
      int64_t num_weights,
      int64_t padding_idx,
      bool scale_grad_by_freq) {
    auto hl_grad = GetOrCreateHbLazyTensor(grad, c10::kHPU);
    hl_grad = HbLazyTensorViews::HandleViewsOrUpdate(grad, hl_grad);
    AddInput(hl_grad.GetIrValue());

    auto hl_indices = GetOrCreateHbLazyTensor(indices, c10::kHPU);
    hl_indices = HbLazyTensorViews::HandleViewsOrUpdate(indices, hl_indices);
    AddInput(hl_indices.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{grad, indices};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        num_weights,
        static_cast<size_t>(EmbeddingBwdParams::NUM_WEIGHTS_INDEX));
    m_meta_data.set(
        padding_idx, static_cast<size_t>(EmbeddingBwdParams::PADDING_INDEX));
    m_meta_data.set(
        scale_grad_by_freq,
        static_cast<size_t>(EmbeddingBwdParams::SCALE_GRADE_BY_FREQ_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", num_weights="
       << m_meta_data.get(
              static_cast<size_t>(EmbeddingBwdParams::NUM_WEIGHTS_INDEX))
       << ", padding_idx="
       << m_meta_data.get(
              static_cast<size_t>(EmbeddingBwdParams::PADDING_INDEX))
       << ", scale_grad_by_freq="
       << m_meta_data.get(static_cast<size_t>(
              EmbeddingBwdParams::SCALE_GRADE_BY_FREQ_INDEX));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
