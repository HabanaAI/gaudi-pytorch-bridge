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

class View : public ir::Node {
 public:
  enum class ViewMeta {
    WEIGHT_INDEX = 1,
    BIAS_INDEX,
    M_INDEX,
    N_INDEX,
    EPS_INDEX
  };
  View() = delete;
  View(const at::Tensor& self, at::IntArrayRef size)
      : Node(
            GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)
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
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
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
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) == 0) {
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

class AsStrided : public ir::Node {
 public:
  enum class AsStridedMeta {
    SIZE_INDEX = 1,
    STRIDE_INDEX = 2,
    STORAGE_OFFSET = 3,
    CAN_REPLACE_OFFSET = 4
  };
  AsStrided() = delete;
  AsStrided(
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

    // additional metadata to signal the replace as_strided pass whether
    // this node can be replaced. Do not replace if there is non-zero storage
    // offset or if self's size doesnt match output size (eg slice)
    // TODO try relaxing these cases for slices by adding additional support in
    // AsStrided PT kernel
    bool can_replace = self.is_contiguous(self.suggest_memory_format());

    auto self_sizes_vec = self.sizes().vec();
    // handle 0-D
    if (self_sizes_vec.size() == 0) {
      self_sizes_vec.emplace_back(1);
    }

    if ((storage_offset) ||
        (prod_sizes(self_sizes_vec) != prod_sizes(size.vec()))) {
      can_replace = false;
    }

    m_meta_data.set(size, static_cast<size_t>(AsStridedMeta::SIZE_INDEX));
    m_meta_data.set(stride, static_cast<size_t>(AsStridedMeta::STRIDE_INDEX));
    m_meta_data.set(
        storage_offset, static_cast<size_t>(AsStridedMeta::STORAGE_OFFSET));
    m_meta_data.set(
        can_replace, static_cast<size_t>(AsStridedMeta::CAN_REPLACE_OFFSET));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", Size = "
       << m_meta_data.get(static_cast<size_t>(AsStridedMeta::SIZE_INDEX))
       << ", strides = "
       << m_meta_data.get(static_cast<size_t>(AsStridedMeta::STRIDE_INDEX))
       << ", storage offset = "
       << m_meta_data.get(static_cast<size_t>(AsStridedMeta::STORAGE_OFFSET))
       << ", can replace = "
       << m_meta_data.get(
              static_cast<size_t>(AsStridedMeta::CAN_REPLACE_OFFSET));
    return ss.str();
  }

  size_t prod_sizes(std::vector<int64_t> sizes) {
    size_t prod_size = 1;
    for (auto s : sizes) {
      prod_size *= s;
    }
    return prod_size;
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
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
      HABANA_ASSERT(m_inputs.size() == 4);
      ss << Node::ToString();
      HABANA_ASSERT(m_inputs[2].DataPtrValidAndNotExpired());
      std::shared_ptr<Data> d1 = m_inputs[2].m_data_ptr.lock();
      ss << ", strides = " << d1->sizes;
      HABANA_ASSERT(m_inputs[3].DataPtrValidAndNotExpired());
      std::shared_ptr<Data> d2 = m_inputs[3].m_data_ptr.lock();
      ss << ", storage offset = " << d2->sizes;
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
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
      HABANA_ASSERT(m_inputs.size() == 4);
      ss << Node::ToString();
      HABANA_ASSERT(m_inputs[1].DataPtrValidAndNotExpired());
      std::shared_ptr<Data> d1 = m_inputs[1].m_data_ptr.lock();
      ss << ", Size = " << d1->sizes;
      HABANA_ASSERT(m_inputs[2].DataPtrValidAndNotExpired());
      std::shared_ptr<Data> d2 = m_inputs[2].m_data_ptr.lock();
      ss << ", strides = " << d2->sizes;
      HABANA_ASSERT(m_inputs[3].DataPtrValidAndNotExpired());
      std::shared_ptr<Data> d3 = m_inputs[3].m_data_ptr.lock();
      ss << ", storage offset = " << d3->sizes;
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

} // namespace ir
} // namespace habana_lazy
