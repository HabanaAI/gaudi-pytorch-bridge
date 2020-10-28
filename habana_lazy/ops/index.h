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

struct Slice : public ir::Node {
  enum class SliceParms {DIM_INDEX=1, START_INDEX, END_INDEX, STEP_INDEX};
  Slice() = delete;
  Slice(const at::Tensor &self, int64_t dim, int64_t start, int64_t end,
    int64_t step) : Node(c10::Symbol::fromQualString("aten::slice")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);

    AddInput(hl_self.GetIrValue());

    m_meta_data.set(dim, static_cast<size_t>(SliceParms::DIM_INDEX));
    m_meta_data.set(start, static_cast<size_t>(SliceParms::START_INDEX));
    m_meta_data.set(end, static_cast<size_t>(SliceParms::END_INDEX));
    m_meta_data.set(step, static_cast<size_t>(SliceParms::STEP_INDEX));

  }

  Slice(const at::Tensor &self, int64_t dim, int64_t index)
    : Node(c10::Symbol::fromQualString("aten::select")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);

    AddInput(hl_self.GetIrValue());

    m_meta_data.set(dim, static_cast<size_t>(SliceParms::DIM_INDEX));
    m_meta_data.set(index, static_cast<size_t>(SliceParms::START_INDEX));

  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim="
      << m_meta_data.get(static_cast<size_t>(SliceParms::DIM_INDEX));

    if ( m_meta_data.count(static_cast<size_t>(SliceParms::END_INDEX)) ) {
      ss << ", start="
        << m_meta_data.get(static_cast<size_t>(SliceParms::START_INDEX))
        << ", end="
        << m_meta_data.get(static_cast<size_t>(SliceParms::END_INDEX))
        << ", step="
        << m_meta_data.get(static_cast<size_t>(SliceParms::STEP_INDEX));
    }
    else {
      ss << ", index="
        << m_meta_data.get(static_cast<size_t>(SliceParms::START_INDEX));
    }

    return ss.str();
  }
};

}
}// habana_lazy