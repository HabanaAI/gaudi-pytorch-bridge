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

class Embedding_forward : public ir::Node {
 public:
  enum class EmbeddingParams {
    PADDING_INDEX = 2,
    SCALE_GRADE_BY_FREQ_INDEX,
    SPARSE_INDEX
  };
  Embedding_forward() = delete;
  Embedding_forward(
      const at::Tensor& weight,
      const at::Tensor& indices,
      int64_t padding_idx,
      bool scale_grad_by_freq,
      bool sparse)
      : Node(c10::Symbol::fromQualString("aten::embedding")) {
    auto hl_weight = GetOrCreateHbLazyTensor(weight, c10::kHABANA);
    AddInput(hl_weight.GetIrValue());

    auto hl_indices = GetOrCreateHbLazyTensor(indices, c10::kHABANA);
    AddInput(hl_indices.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{weight, indices};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        padding_idx, static_cast<size_t>(EmbeddingParams::PADDING_INDEX));
    m_meta_data.set(
        scale_grad_by_freq,
        static_cast<size_t>(EmbeddingParams::SCALE_GRADE_BY_FREQ_INDEX));
    m_meta_data.set(sparse, static_cast<size_t>(EmbeddingParams::SPARSE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", padding_idx="
       << m_meta_data.get(static_cast<size_t>(EmbeddingParams::PADDING_INDEX))
       << ", scale_grad_by_freq="
       << m_meta_data.get(
              static_cast<size_t>(EmbeddingParams::SCALE_GRADE_BY_FREQ_INDEX))
       << ", sparse="
       << m_meta_data.get(static_cast<size_t>(EmbeddingParams::SPARSE_INDEX));
    return ss.str();
  }
};

class Embedding_backward : public ir::Node {
 public:
  enum class EmbeddingBwdParams {
    NUM_WEIGHTS_INDEX = 2,
    PADDING_INDEX,
    SCALE_GRADE_BY_FREQ_INDEX
  };
  Embedding_backward() = delete;
  Embedding_backward(
      const at::Tensor& grad,
      const at::Tensor& indices,
      int64_t num_weights,
      int64_t padding_idx,
      bool scale_grad_by_freq)
      : Node(c10::Symbol::fromQualString("aten::embedding_dense_backward")) {
    auto hl_grad = GetOrCreateHbLazyTensor(grad, c10::kHABANA);
    AddInput(hl_grad.GetIrValue());

    auto hl_indices = GetOrCreateHbLazyTensor(indices, c10::kHABANA);
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
