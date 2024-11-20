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
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"
namespace habana_lazy {
namespace ir {

class View : public ir::Node {
 public:
  View() = delete;
  View(const at::Tensor& self, at::IntArrayRef size)
      : Node(
            habana_helpers::GetRefineDynamicShapeStatus()
                ? c10::Symbol::fromQualString("hpu::view")
                : c10::Symbol::fromQualString("aten::view")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);
    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    /*
     * For Dynamic Shape, in case of view tensor the constant is
     * converted to shape tensor. and added as input & hence we do
     * not set the meta data here.
     */
    if (habana_helpers::GetRefineDynamicShapeStatus()) {
      auto shape = empty_hpu_lazy(
          size,
          self.options(),
          c10::MemoryFormat::Contiguous,
          false,
          SHAPE_TENSOR);
      auto hl_shape = GetOrCreateHbLazyTensor(shape, c10::kHPU);
      AddInput(hl_shape.GetIrValue());
      input_pt_vec.emplace_back(shape);
    } else {
      m_meta_data.set(size, static_cast<size_t>(1));
    }
    AddInputPtTensors(input_pt_vec);
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString();
    if (habana_helpers::GetRefineDynamicShapeStatus() == 0) {
      ss << ", View Size = " << m_meta_data.get(static_cast<size_t>(1));
    } else {
      HABANA_ASSERT(m_inputs.size() == 2);
      auto& shape = m_inputs[1];
      if (shape.DataPtrValidAndNotExpired()) {
        std::shared_ptr<Data> data = shape.m_data_ptr.lock();
        ss << ", View Size = " << data->sizes;
      }
    }
    return ss.str();
  }
};

class StridedInsert : public ir::Node {
 public:
  enum class StridedInsertMeta { STRIDE_INDEX = 2, STORAGE_OFFSET = 3 };
  StridedInsert() = delete;
  StridedInsert(
      const at::Tensor& orig_t,
      const at::Tensor& insert_t,
      at::IntArrayRef stride,
      int64_t storage_offset,
      std::string node_str)
      : Node(c10::Symbol::fromQualString(node_str)) {
    auto hl_orig = habana_lazy::GetOrCreateHbLazyTensor(orig_t, c10::kHPU);
    AddInput(hl_orig.GetIrValue());

    auto hl_insert = habana_lazy::GetOrCreateHbLazyTensor(insert_t, c10::kHPU);
    AddInput(hl_insert.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{orig_t, insert_t};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        stride, static_cast<size_t>(StridedInsertMeta::STRIDE_INDEX));
    m_meta_data.set(
        storage_offset, static_cast<size_t>(StridedInsertMeta::STORAGE_OFFSET));
  }

  StridedInsert(
      const at::Tensor& orig_t,
      const at::Tensor& insert_t,
      const at::Tensor& storage_offset_t,
      std::string node_str)
      : Node(c10::Symbol::fromQualString(node_str)) {
    auto hl_orig = habana_lazy::GetOrCreateHbLazyTensor(orig_t, c10::kHPU);
    AddInput(hl_orig.GetIrValue());

    auto hl_insert = habana_lazy::GetOrCreateHbLazyTensor(insert_t, c10::kHPU);
    AddInput(hl_insert.GetIrValue());

    auto hl_storage_offset =
        habana_lazy::GetOrCreateHbLazyTensor(storage_offset_t, c10::kHPU);
    AddInput(hl_storage_offset.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{orig_t, insert_t, storage_offset_t};
    AddInputPtTensors(input_pt_vec);
  }

  StridedInsert(
      const at::Tensor& orig_t,
      const at::Tensor& insert_t,
      const at::Tensor& stride_t,
      const at::Tensor& storage_offset_t,
      std::string node_str)
      : Node(c10::Symbol::fromQualString(node_str)) {
    auto hl_orig = habana_lazy::GetOrCreateHbLazyTensor(orig_t, c10::kHPU);
    AddInput(hl_orig.GetIrValue());

    auto hl_insert = habana_lazy::GetOrCreateHbLazyTensor(insert_t, c10::kHPU);
    AddInput(hl_insert.GetIrValue());

    auto hl_stride = habana_lazy::GetOrCreateHbLazyTensor(stride_t, c10::kHPU);
    AddInput(hl_stride.GetIrValue());

    auto hl_storage_offset =
        habana_lazy::GetOrCreateHbLazyTensor(storage_offset_t, c10::kHPU);
    AddInput(hl_storage_offset.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{
        orig_t, insert_t, stride_t, storage_offset_t};
    AddInputPtTensors(input_pt_vec);
  }

  std::string ToString() const override {
    std::stringstream ss;
    if (habana_helpers::GetRefineDynamicShapeStatus()) {
      ss << Node::ToString();
      if (m_inputs.size() == 3) {
        HABANA_ASSERT(m_inputs[2].DataPtrValidAndNotExpired());
        std::shared_ptr<Data> d2 = m_inputs[2].m_data_ptr.lock();
        ss << ", storage offset = " << d2->sizes;
      } else {
        HABANA_ASSERT(m_inputs.size() == 4);
        HABANA_ASSERT(m_inputs[2].DataPtrValidAndNotExpired());
        std::shared_ptr<Data> d1 = m_inputs[2].m_data_ptr.lock();
        ss << ", strides = " << d1->sizes;
        HABANA_ASSERT(m_inputs[3].DataPtrValidAndNotExpired());
        std::shared_ptr<Data> d2 = m_inputs[3].m_data_ptr.lock();
        ss << ", storage offset = " << d2->sizes;
      }
    } else {
      ss << Node::ToString() << ", Strides = "
         << m_meta_data.get(
                static_cast<size_t>(StridedInsertMeta::STRIDE_INDEX))
         << ", storage offset = "
         << m_meta_data.get(
                static_cast<size_t>(StridedInsertMeta::STORAGE_OFFSET));
    }
    return ss.str();
  }
};

struct SliceInsert : public ir::Node {
  enum class SliceInsertParams { PARAMS_INDEX = 2 };
  SliceInsert() = delete;
  SliceInsert(
      const at::Tensor& orig_t,
      const at::Tensor& insert_t,
      at::IntArrayRef params)
      : Node(
            habana_helpers::GetRefineDynamicShapeStatus()
                ? GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_SLICE)
                    ? c10::Symbol::fromQualString("hpu::slice_insert_ds_ht")
                    : c10::Symbol::fromQualString("hpu::slice_insert_ds")
                : c10::Symbol::fromQualString("hpu::slice_insert")) {
    auto hl_orig = habana_lazy::GetOrCreateHbLazyTensor(orig_t, c10::kHPU);
    AddInput(hl_orig.GetIrValue());

    auto hl_insert = habana_lazy::GetOrCreateHbLazyTensor(insert_t, c10::kHPU);
    AddInput(hl_insert.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{orig_t, insert_t};

    if (habana_helpers::GetRefineDynamicShapeStatus()) {
      if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_SLICE)) {
        std::vector<int64_t> host_params{
            orig_t.dim(), 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
        int num_slice_params = params.size() / 4;
        int index = 0;

        for (int i = 0; i < num_slice_params; i++) {
          int64_t dim = params[i * 4];
          int64_t start = params[i * 4 + 1];
          int64_t end = params[i * 4 + 2];
          int64_t step = params[i * 4 + 3];

          // one place to wrap all dim, start and end indicies
          habana::SliceOperator::compute_output_shape(
              orig_t, dim, start, end, step);
          index = orig_t.dim() - dim;
          host_params[index] = step;
          host_params[index + 5] = start;
        }
        auto host_tensor = empty_hpu_lazy(
            host_params.size() * 2,
            orig_t.options(),
            orig_t.suggest_memory_format(),
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
        auto dims = orig_t.dim();
        std::vector<int64_t> step_vec(dims, 1);
        std::vector<int64_t> start_vec(dims, 0);

        int num_slice_params = params.size() / 4;
        for (int i = 0; i < num_slice_params; i++) {
          int64_t dim = params[i * 4];
          int64_t start = params[i * 4 + 1];
          int64_t end = params[i * 4 + 2];
          int64_t step = params[i * 4 + 3];

          // one place to wrap all dim, start and end indicies
          habana::SliceOperator::compute_output_shape(
              orig_t, dim, start, end, step);
          start_vec[dim] = start;
          step_vec[dim] = step;
        }
        auto step_t = empty_hpu_lazy(
            c10::IntArrayRef(step_vec.data(), step_vec.size()),
            orig_t.options(),
            c10::MemoryFormat::Contiguous,
            false,
            SHAPE_TENSOR);
        auto hl_step = GetOrCreateHbLazyTensor(step_t, c10::kHPU);
        AddInput(hl_step.GetIrValue());
        input_pt_vec.emplace_back(step_t);
        auto start_t = empty_hpu_lazy(
            c10::IntArrayRef(start_vec.data(), start_vec.size()),
            orig_t.options(),
            c10::MemoryFormat::Contiguous,
            false,
            SHAPE_TENSOR);
        auto hl_start = GetOrCreateHbLazyTensor(start_t, c10::kHPU);
        AddInput(hl_start.GetIrValue());
        input_pt_vec.emplace_back(start_t);
      }
    } else {
      m_meta_data.set(
          params, static_cast<size_t>(SliceInsertParams::PARAMS_INDEX));
    }
    AddInputPtTensors(input_pt_vec);
  }

  std::string ToString() const override {
    std::stringstream ss;
    if (habana_helpers::GetRefineDynamicShapeStatus()) {
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
      ss << Node::ToString() << ", slice_params(dim, start, end, step)="
         << m_meta_data.get(
                static_cast<size_t>(SliceInsertParams::PARAMS_INDEX));
    }
    return ss.str();
  }
};
class StridedView : public ir::Node {
 public:
  enum class StridedViewMeta {
    SIZE_INDEX = 1,
    STRIDE_INDEX = 2,
    STORAGE_OFFSET = 3
  };
  StridedView() = delete;
  StridedView(
      const at::Tensor& self,
      at::IntArrayRef size,
      at::IntArrayRef stride,
      int64_t storage_offset,
      std::string node_str)
      : Node(c10::Symbol::fromQualString(node_str)) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(size, static_cast<size_t>(StridedViewMeta::SIZE_INDEX));
    m_meta_data.set(stride, static_cast<size_t>(StridedViewMeta::STRIDE_INDEX));
    m_meta_data.set(
        storage_offset, static_cast<size_t>(StridedViewMeta::STORAGE_OFFSET));
  }

  StridedView(
      const at::Tensor& self,
      at::Tensor& size,
      at::Tensor& storage_offset,
      std::string node_str)
      : Node(c10::Symbol::fromQualString(node_str)) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    auto hl_size = habana_lazy::GetOrCreateHbLazyTensor(size, c10::kHPU);
    AddInput(hl_size.GetIrValue());

    auto hl_storage_offset =
        habana_lazy::GetOrCreateHbLazyTensor(storage_offset, c10::kHPU);
    AddInput(hl_storage_offset.GetIrValue());
    std::vector<at::Tensor> input_pt_vec{self, size, storage_offset};
    AddInputPtTensors(input_pt_vec);
  }

  StridedView(
      const at::Tensor& self,
      at::Tensor& size,
      at::Tensor& stride,
      at::Tensor& storage_offset,
      std::string node_str)
      : Node(c10::Symbol::fromQualString(node_str)) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    auto hl_size = habana_lazy::GetOrCreateHbLazyTensor(size, c10::kHPU);
    AddInput(hl_size.GetIrValue());

    auto hl_stride = habana_lazy::GetOrCreateHbLazyTensor(stride, c10::kHPU);
    AddInput(hl_stride.GetIrValue());

    auto hl_storage_offset =
        habana_lazy::GetOrCreateHbLazyTensor(storage_offset, c10::kHPU);
    AddInput(hl_storage_offset.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self, size, stride, storage_offset};
    AddInputPtTensors(input_pt_vec);
  }

  std::string ToString() const override {
    std::stringstream ss;
    if (habana_helpers::GetRefineDynamicShapeStatus()) {
      ss << Node::ToString();
      HABANA_ASSERT(m_inputs[1].DataPtrValidAndNotExpired());
      std::shared_ptr<Data> d1 = m_inputs[1].m_data_ptr.lock();
      ss << ", Size = " << d1->sizes;
      // if input size is 4 strides are part of frontend
      if (m_inputs.size() == 4) {
        HABANA_ASSERT(m_inputs[2].DataPtrValidAndNotExpired());
        std::shared_ptr<Data> d2 = m_inputs[2].m_data_ptr.lock();
        ss << ", strides = " << d2->sizes;
        HABANA_ASSERT(m_inputs[3].DataPtrValidAndNotExpired());
        std::shared_ptr<Data> d3 = m_inputs[3].m_data_ptr.lock();
        ss << ", storage offset = " << d3->sizes;
        // if input size is 3 strides are part of implementor
      } else {
        HABANA_ASSERT(m_inputs[2].DataPtrValidAndNotExpired());
        std::shared_ptr<Data> d2 = m_inputs[2].m_data_ptr.lock();
        ss << ", storage offset = " << d2->sizes;
      }
    } else {
      ss << Node::ToString() << ", Size = "
         << m_meta_data.get(static_cast<size_t>(StridedViewMeta::SIZE_INDEX))
         << ", strides = "
         << m_meta_data.get(static_cast<size_t>(StridedViewMeta::STRIDE_INDEX))
         << ", storage offset = ";
    }
    return ss.str();
  }
};

class Expand : public ir::Node {
 public:
  Expand() = delete;
  Expand(const at::Tensor& self, at::IntArrayRef sizes, bool implicit)
      : Node(c10::Symbol::fromQualString("hpu::expand")) {
    auto hl_self = habana_lazy::GetHbLazyTensor(self);
    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    m_meta_data.set(sizes, static_cast<size_t>(1));
    m_meta_data.set(implicit, static_cast<size_t>(2));
    AddInputPtTensors(input_pt_vec);
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString();
    ss << ", Expand Size = " << m_meta_data.get(static_cast<size_t>(1));
    ss << ", implicit = " << m_meta_data.get(static_cast<size_t>(2));
    return ss.str();
  }
};

} // namespace ir
} // namespace habana_lazy
