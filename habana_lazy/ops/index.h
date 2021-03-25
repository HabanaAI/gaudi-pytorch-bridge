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
  enum class SliceParms { DIM_INDEX = 1, START_INDEX, END_INDEX, STEP_INDEX };
  Slice() = delete;
  Slice(
      const at::Tensor& self,
      int64_t dim,
      int64_t start,
      int64_t end,
      int64_t step)
      : Node(c10::Symbol::fromQualString("aten::slice")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, static_cast<size_t>(SliceParms::DIM_INDEX));
    m_meta_data.set(start, static_cast<size_t>(SliceParms::START_INDEX));
    m_meta_data.set(end, static_cast<size_t>(SliceParms::END_INDEX));
    m_meta_data.set(step, static_cast<size_t>(SliceParms::STEP_INDEX));
  }

  Slice(const at::Tensor& self, int64_t dim, int64_t index)
      : Node(c10::Symbol::fromQualString("aten::select")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, static_cast<size_t>(SliceParms::DIM_INDEX));
    m_meta_data.set(index, static_cast<size_t>(SliceParms::START_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim="
       << m_meta_data.get(static_cast<size_t>(SliceParms::DIM_INDEX));

    if (m_meta_data.count(static_cast<size_t>(SliceParms::END_INDEX))) {
      ss << ", start="
         << m_meta_data.get(static_cast<size_t>(SliceParms::START_INDEX))
         << ", end="
         << m_meta_data.get(static_cast<size_t>(SliceParms::END_INDEX))
         << ", step="
         << m_meta_data.get(static_cast<size_t>(SliceParms::STEP_INDEX));
    } else {
      ss << ", index="
         << m_meta_data.get(static_cast<size_t>(SliceParms::START_INDEX));
    }

    return ss.str();
  }
};

struct IndexSelect : public ir::Node {
  enum class IndexSelectParams { DIM_INDEX = 1 };
  IndexSelect() = delete;
  IndexSelect(const at::Tensor& self, int64_t dim, const Tensor& index)
      : Node(c10::Symbol::fromQualString("aten::index_select")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto hl_index = habana_lazy::GetOrCreateHbLazyTensor(index, c10::kHABANA);

    AddInput(hl_self.GetIrValue());
    AddInput(hl_index.GetIrValue());

    m_meta_data.set(dim, static_cast<size_t>(IndexSelectParams::DIM_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim="
       << m_meta_data.get(static_cast<size_t>(IndexSelectParams::DIM_INDEX));

    return ss.str();
  }
};

struct IndexAdd_ : public ir::Node {
  enum class IndexAdd_Params { DIM_INDEX = 1 };
  IndexAdd_() = delete;
  IndexAdd_(
      at::Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& source)
      : Node(c10::Symbol::fromQualString("aten::index_add")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto hl_index = habana_lazy::GetOrCreateHbLazyTensor(index, c10::kHABANA);
    auto hl_source = habana_lazy::GetOrCreateHbLazyTensor(source, c10::kHABANA);

    AddInput(hl_self.GetIrValue());
    AddInput(hl_index.GetIrValue());
    AddInput(hl_source.GetIrValue());

    m_meta_data.set(dim, static_cast<size_t>(IndexAdd_Params::DIM_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim="
       << m_meta_data.get(static_cast<size_t>(IndexAdd_Params::DIM_INDEX));

    return ss.str();
  }
};

struct ScatterValue : public ir::Node {
  enum class ScatterValue_Params { DIM_INDEX = 1 };
  ScatterValue() = delete;
  ScatterValue(at::Tensor& self, int64_t dim, const Tensor& index, Scalar value)
      : Node(c10::Symbol::fromQualString("hpu::scatter_value")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto hl_index = habana_lazy::GetOrCreateHbLazyTensor(index, c10::kHABANA);
    auto hl_value = habana_lazy::GetIrValueForScalar(value);

    AddInput(hl_self.GetIrValue());
    AddInput(hl_index.GetIrValue());
    AddInput(hl_value);
    std::vector<at::Tensor> input_pt_vec{self, index};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, static_cast<size_t>(ScatterValue_Params::DIM_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim="
       << m_meta_data.get(static_cast<size_t>(ScatterValue_Params::DIM_INDEX));

    return ss.str();
  }
};

} // namespace ir
} // namespace habana_lazy
