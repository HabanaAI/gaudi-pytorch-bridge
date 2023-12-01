/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
            habana_helpers::GetRefineDynamicShapeStatus()
                ? GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_SLICE)
                    ? c10::Symbol::fromQualString("hpu::slice_ht")
                    : c10::Symbol::fromQualString("hpu::slice")
                : c10::Symbol::fromQualString("aten::slice")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);

    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    /*
     * For Dynamic Shape, in case of view tensor the start/step constant is
     * converted to shape tensor. and added as input & hence we do
     * not set the meta data here.
     */
    if (habana_helpers::GetRefineDynamicShapeStatus()) {
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

      if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_SLICE)) {
        std::vector<int64_t> host_params{
            self.dim(), 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
        int index = self.dim() - dim;
        host_params[index] = step;
        host_params[index + 5] = start;
        auto host_tensor = empty_hpu_lazy(
            host_params.size() * 2,
            self.options(),
            self.suggest_memory_format(),
            false,
            HOST_TO_DEVICE_TENSOR);
        auto hl_param_tensor = GetOrCreateHbLazyTensor(host_tensor, c10::kHPU);
        auto hl_param_tensor_internal =
            hl_param_tensor.CurrentTensorAttached().value();
        auto host_tmeta{
            habana::get_tensor_extra_meta(hl_param_tensor_internal)};
        host_tmeta->set_host_data(
            host_params.data(),
            host_params.size(),
            sizeof(uint64_t),
            habana::HostDataType::UINT64_T);
        host_tmeta->set_H2D_data_for_bucketing();
        AddInput(hl_param_tensor.GetIrValue());
        input_pt_vec.emplace_back(host_tensor);
      } else {
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
      }
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

    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, static_cast<size_t>(SliceParms::DIM_INDEX));
    m_meta_data.set(index, static_cast<size_t>(SliceParms::START_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    if (habana_helpers::GetRefineDynamicShapeStatus()) {
      auto& shape = m_inputs[1];
      HABANA_ASSERT(shape.DataPtrValidAndNotExpired());
      std::shared_ptr<Data> data_shape = shape.m_data_ptr.lock();
      ss << ", shape =" << data_shape->sizes;
      if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_SLICE)) {
        auto& params = m_inputs[2];
        HABANA_ASSERT(params.DataPtrValidAndNotExpired());
        std::shared_ptr<Data> params_ = params.m_data_ptr.lock();
        ss << ", params=" << params_->sizes;
      } else {
        auto& start = m_inputs[3];
        HABANA_ASSERT(start.DataPtrValidAndNotExpired());
        std::shared_ptr<Data> data_start = start.m_data_ptr.lock();
        ss << ", start=" << data_start->sizes;
        auto& step = m_inputs[2];
        HABANA_ASSERT(step.DataPtrValidAndNotExpired());
        std::shared_ptr<Data> data_step = step.m_data_ptr.lock();
        ss << ", step=" << data_step->sizes;
      }
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

struct ScatterAdd : public ir::Node {
  enum class ScatterAdd_Params { DIM_INDEX = 1 };
  ScatterAdd() = delete;
  ScatterAdd(
      at::Tensor& self,
      int64_t dim,
      const at::Tensor& index,
      const at::Tensor& src)
      : Node(c10::Symbol::fromQualString("hpu::scatter_add")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);
    auto hl_index = habana_lazy::GetOrCreateHbLazyTensor(index, c10::kHPU);
    auto hl_src = habana_lazy::GetOrCreateHbLazyTensor(src, c10::kHPU);

    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);
    hl_index = HbLazyTensorViews::HandleViewsOrUpdate(index, hl_index);
    hl_src = HbLazyTensorViews::HandleViewsOrUpdate(src, hl_src);

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

struct SqueezeBase : public ir::Node {
  enum class SqueezeParams { DIM_INDEX = 1 };
  SqueezeBase() = delete;
  SqueezeBase(const at::Tensor& self, int64_t dim, std::string node_str)
      : Node(c10::Symbol::fromQualString(node_str)) {
    auto hl_self = habana_lazy::GetHbLazyTensor(self);

    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};

    m_meta_data.set(dim, static_cast<size_t>(SqueezeParams::DIM_INDEX));
    AddInputPtTensors(input_pt_vec);
  }
};

struct SqueezeDims : public ir::Node {
  enum class SqueezeParams { DIM_INDEX = 1 };
  SqueezeDims() = delete;
  SqueezeDims(
      const at::Tensor& self,
      at::IntArrayRef dims,
      std::string node_str = "aten::squeeze")
      : Node(c10::Symbol::fromQualString(node_str)) {
    auto hl_self = habana_lazy::GetHbLazyTensor(self);

    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};

    m_meta_data.set(dims, static_cast<size_t>(SqueezeParams::DIM_INDEX));
    AddInputPtTensors(input_pt_vec);
  }
};

struct Identity : public ir::Node {
  Identity() = delete;
  Identity(const at::Tensor& self, std::string node_str)
      : Node(c10::Symbol::fromQualString(node_str)) {
    auto hl_self = habana_lazy::GetHbLazyTensor(self);

    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);

    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);
  }
};

} // namespace ir
} // namespace habana_lazy
