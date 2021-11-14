/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
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

class TopK : public ir::Node {
 public:
  enum class TopKParams { K_INDEX = 1, DIM_INDEX, LARGEST_INDEX, SORTED_INDEX };
  TopK() = delete;
  TopK(
      const at::Tensor& self,
      int64_t k,
      int64_t dim,
      bool largest,
      bool sorted)
      : Node(
            GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)
                ? c10::Symbol::fromQualString("hpu::topk")
                : c10::Symbol::fromQualString("aten::topk")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};

    if (GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
      auto input_shape = empty_hpu_lazy(
          k, self.options(), self.suggest_memory_format(), false, SHAPE_TENSOR);
      auto hl_input_shape = GetOrCreateHbLazyTensor(input_shape, c10::kHPU);
      AddInput(hl_input_shape.GetIrValue());
      input_pt_vec.emplace_back(input_shape);
    } else {
      m_meta_data.set(k, static_cast<size_t>(TopKParams::K_INDEX));
    }

    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, static_cast<size_t>(TopKParams::DIM_INDEX));
    m_meta_data.set(largest, static_cast<size_t>(TopKParams::LARGEST_INDEX));
    m_meta_data.set(sorted, static_cast<size_t>(TopKParams::SORTED_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString();
    if (GET_ENV_FLAG(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
      auto& input_shape_k = m_inputs[1];
      if (input_shape_k.DataPtrValidAndNotExpired()) {
        std::shared_ptr<Data> data_k = input_shape_k.m_data_ptr.lock();
        ss << ", k = " << data_k->sizes;
      }
    } else {
      ss << ", k = "
         << m_meta_data.get(static_cast<size_t>(TopKParams::K_INDEX));
    }

    ss << ", dim = "
       << m_meta_data.get(static_cast<size_t>(TopKParams::DIM_INDEX))
       << ", largest = "
       << m_meta_data.get(static_cast<size_t>(TopKParams::LARGEST_INDEX))
       << ", sorted = "
       << m_meta_data.get(static_cast<size_t>(TopKParams::SORTED_INDEX));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
