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
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);

    hl_self = HandleViewsOrUpdate(self, hl_self);

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
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);

    hl_self = HandleViewsOrUpdate(self, hl_self);

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
  IndexSelect(const at::Tensor& self, int64_t dim, const at::Tensor& index)
      : Node(c10::Symbol::fromQualString("aten::index_select")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);
    auto hl_index = habana_lazy::GetOrCreateHbLazyTensor(index, c10::kHPU);

    hl_self = HandleViewsOrUpdate(self, hl_self);
    hl_index = HandleViewsOrUpdate(index, hl_index);

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

struct ScatterAdd : public ir::Node {
  enum class ScatterAdd_Params { DIM_INDEX = 1 };
  ScatterAdd() = delete;
  ScatterAdd(
      at::Tensor& self,
      int64_t dim,
      const at::Tensor& index,
      const at::Tensor& src)
      : Node(c10::Symbol::fromQualString("aten::scatter_add")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);
    auto hl_index = habana_lazy::GetOrCreateHbLazyTensor(index, c10::kHPU);
    auto hl_src = habana_lazy::GetOrCreateHbLazyTensor(src, c10::kHPU);

    hl_self = HandleViewsOrUpdate(self, hl_self);
    hl_index = HandleViewsOrUpdate(index, hl_index);
    hl_src = HandleViewsOrUpdate(src, hl_src);

    AddInput(hl_self.GetIrValue());
    AddInput(hl_index.GetIrValue());
    AddInput(hl_src.GetIrValue());
    std::vector<at::Tensor> input_pt_vec{self, index, src};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, static_cast<size_t>(ScatterAdd_Params::DIM_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim="
       << m_meta_data.get(static_cast<size_t>(ScatterAdd_Params::DIM_INDEX));

    return ss.str();
  }
};

} // namespace ir
} // namespace habana_lazy
