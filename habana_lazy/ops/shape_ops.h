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
      : Node(c10::Symbol::fromQualString("aten::view")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(size, static_cast<size_t>(1));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString()
       << ", View Size = " << m_meta_data.get(static_cast<size_t>(1));
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
} // namespace ir
} // namespace habana_lazy
