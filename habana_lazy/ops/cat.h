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

class Cat : public Node {
 public:
  enum class CatIndex { kDimIdx = 1 };
  Cat() = delete;
  Cat(const at::TensorList tensors, int64_t dim_)
      : Node(c10::Symbol::fromQualString("aten::cat")) {
    ValueList hl_tensors;
    std::vector<at::Tensor> input_pt_vec;
    for (auto& i : tensors) {
      auto hl_tensor = GetOrCreateHbLazyTensor(i, c10::kHPU);
      hl_tensors.push_back(hl_tensor.GetIrValue());
      input_pt_vec.emplace_back(i);
    }

    auto cat_input = GetIrValueForListConstruct(hl_tensors);
    cat_input.mp_node->AddInputPtTensors(input_pt_vec);
    AddInput(cat_input);
    m_meta_data.set(dim_, static_cast<size_t>(CatIndex::kDimIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString()
       << ", dim=" << m_meta_data.get(static_cast<size_t>(CatIndex::kDimIdx));
    return ss.str();
  }
};

class SplitWithSize : public ir::Node {
 public:
  enum class SplitSizeIdx { kSplitSizesIdx = 1, kDimIdx };
  SplitWithSize() = delete;
  SplitWithSize(
      const at::Tensor& self,
      c10::IntArrayRef split_sizes,
      int64_t dim)
      : Node(c10::Symbol::fromQualString("aten::split_with_sizes")) {
    auto hl_self = GetHbLazyTensor(self);
    hl_self = HandleViewsOrUpdate(self, hl_self);
    AddInput(hl_self.GetIrValue());
    m_meta_data.set(
        split_sizes, static_cast<size_t>(SplitSizeIdx::kSplitSizesIdx));
    m_meta_data.set(dim, static_cast<size_t>(SplitSizeIdx::kDimIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", split_sizes="
       << m_meta_data.get(static_cast<size_t>(SplitSizeIdx::kSplitSizesIdx))
       << ", dim="
       << m_meta_data.get(static_cast<size_t>(SplitSizeIdx::kDimIdx));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
