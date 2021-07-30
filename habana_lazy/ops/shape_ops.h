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
  enum class AsStridedMeta { SIZE_INDEX = 1, STRIDE_INDEX = 2, STORAGE_OFFSET };
  AsStrided() = delete;
  AsStrided(
      const at::Tensor& self,
      at::IntArrayRef size,
      at::IntArrayRef stride,
      int64_t storage_offset)
      : Node(c10::Symbol::fromQualString("hpu::as_strided_lazy_")) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(size, static_cast<size_t>(AsStridedMeta::SIZE_INDEX));
    m_meta_data.set(stride, static_cast<size_t>(AsStridedMeta::STRIDE_INDEX));
    m_meta_data.set(
        storage_offset, static_cast<size_t>(AsStridedMeta::STORAGE_OFFSET));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", Size = "
       << m_meta_data.get(static_cast<size_t>(AsStridedMeta::SIZE_INDEX))
       << ", strides = "
       << m_meta_data.get(static_cast<size_t>(AsStridedMeta::STRIDE_INDEX))
       << ", storage offset = "
       << m_meta_data.get(static_cast<size_t>(AsStridedMeta::STORAGE_OFFSET));
    return ss.str();
  }
};

} // namespace ir
} // namespace habana_lazy
