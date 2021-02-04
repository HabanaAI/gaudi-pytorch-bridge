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
      : Node(c10::Symbol::fromQualString("aten::topk")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(k, static_cast<size_t>(TopKParams::K_INDEX));
    m_meta_data.set(dim, static_cast<size_t>(TopKParams::DIM_INDEX));
    m_meta_data.set(largest, static_cast<size_t>(TopKParams::LARGEST_INDEX));
    m_meta_data.set(sorted, static_cast<size_t>(TopKParams::SORTED_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString()
       << ", k = " << m_meta_data.get(static_cast<size_t>(TopKParams::K_INDEX))
       << ", dim = "
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
