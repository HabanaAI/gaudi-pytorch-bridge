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
enum class SumIndex { kDtypIdx = 1 };
class Sum : public Node {
 public:
  Sum() = delete;
  Sum(const Tensor& self, c10::optional<ScalarType> dtype)
      : Node(c10::Symbol::fromQualString("aten::sum")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto ir_value = hl_self.GetIrValue();
    AddInput(ir_value);

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dtype, static_cast<size_t>(SumIndex::kDtypIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dtype="
       << m_meta_data.get(static_cast<size_t>(SumIndex::kDtypIdx));
    return ss.str();
  }
};

enum class SumDimIntListIndex { kDimIdx = 1, kKeepdimIdx = 2, kDtypeIdx = 3 };
class SumDimIntList : public Node {
 public:
  SumDimIntList() = delete;
  SumDimIntList(
      const Tensor& self,
      IntArrayRef dim,
      bool keepdim,
      c10::optional<ScalarType> dtype)
      : Node(c10::Symbol::fromQualString("hpu::sum_dim_IntList")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto ir_value = hl_self.GetIrValue();
    AddInput(ir_value);

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, static_cast<size_t>(SumDimIntListIndex::kDimIdx));
    m_meta_data.set(
        keepdim, static_cast<size_t>(SumDimIntListIndex::kKeepdimIdx));
    m_meta_data.set(dtype, static_cast<size_t>(SumDimIntListIndex::kDtypeIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim="
       << m_meta_data.get(static_cast<size_t>(SumDimIntListIndex::kDimIdx))
       << ", keepdim="
       << m_meta_data.get(static_cast<size_t>(SumDimIntListIndex::kKeepdimIdx))
       << ", dtype="
       << m_meta_data.get(static_cast<size_t>(SumDimIntListIndex::kDtypeIdx));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
