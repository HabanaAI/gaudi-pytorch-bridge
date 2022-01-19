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
#include "habana_kernels/index_kernels.h"
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
      : Node(
            GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)
                ? c10::Symbol::fromQualString("hpu::slice")
                : c10::Symbol::fromQualString("aten::slice")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);

    hl_self = HandleViewsOrUpdate(self, hl_self);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    /*
     * For Dynamic Shape, in case of view tensor the start/step constant is
     * converted to shape tensor. and added as input & hence we do
     * not set the meta data here.
     */
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
      dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

      end = self.sizes().vec()[dim] < end ? self.sizes().vec()[dim] : end;
      auto shape = habana::SliceOperator::compute_output_shape(
          self, dim, start, end, step);
      auto shape_t = empty_hpu_lazy(
          shape,
          self.options(),
          c10::MemoryFormat::Contiguous,
          false,
          SHAPE_TENSOR);
      auto hl_shape = GetOrCreateHbLazyTensor(shape_t, c10::kHPU);
      AddInput(hl_shape.GetIrValue());
      input_pt_vec.emplace_back(shape_t);
      auto dims = self.dim();
      std::vector<int64_t> step_vec(dims, 1);
      step_vec[dim] = step;
      auto step_t = empty_hpu_lazy(
          c10::IntArrayRef(step_vec.data(), step_vec.size()),
          self.options(),
          c10::MemoryFormat::Contiguous,
          false,
          SHAPE_TENSOR);
      auto hl_step = GetOrCreateHbLazyTensor(step_t, c10::kHPU);
      AddInput(hl_step.GetIrValue());
      input_pt_vec.emplace_back(step_t);
      std::vector<int64_t> start_vec(dims, 0);
      start_vec[dim] = start;
      auto start_t = empty_hpu_lazy(
          c10::IntArrayRef(start_vec.data(), start_vec.size()),
          self.options(),
          c10::MemoryFormat::Contiguous,
          false,
          SHAPE_TENSOR);
      auto hl_start = GetOrCreateHbLazyTensor(start_t, c10::kHPU);
      AddInput(hl_start.GetIrValue());
      input_pt_vec.emplace_back(start_t);
    } else {
      m_meta_data.set(dim, static_cast<size_t>(SliceParms::DIM_INDEX));
      m_meta_data.set(start, static_cast<size_t>(SliceParms::START_INDEX));
      m_meta_data.set(step, static_cast<size_t>(SliceParms::STEP_INDEX));
      m_meta_data.set(end, static_cast<size_t>(SliceParms::END_INDEX));
    }
    AddInputPtTensors(input_pt_vec);
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
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) &&
        (m_inputs.size() == 4)) {
      auto& shape = m_inputs[1];
      HABANA_ASSERT(shape.DataPtrValidAndNotExpired());
      std::shared_ptr<Data> data_shape = shape.m_data_ptr.lock();
      ss << ", shape =" << data_shape->sizes;
      auto& start = m_inputs[3];
      HABANA_ASSERT(start.DataPtrValidAndNotExpired());
      std::shared_ptr<Data> data_start = start.m_data_ptr.lock();
      ss << ", start=" << data_start->sizes;
      auto& step = m_inputs[2];
      HABANA_ASSERT(step.DataPtrValidAndNotExpired());
      std::shared_ptr<Data> data_step = step.m_data_ptr.lock();
      ss << ", step=" << data_step->sizes;
    } else {
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
