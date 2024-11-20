/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
