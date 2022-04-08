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
enum class SumIndex { kDtypIdx = 1 };

enum class ProdIndex { kDtypIdx = 1 };
class Prod : public Node {
 public:
  Prod() = delete;
  Prod(const at::Tensor& self, c10::optional<at::ScalarType> dtype)
      : Node(c10::Symbol::fromQualString("aten::prod")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);
    auto ir_value = hl_self.GetIrValue();
    AddInput(ir_value);

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dtype, static_cast<size_t>(ProdIndex::kDtypIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dtype="
       << m_meta_data.get(static_cast<size_t>(ProdIndex::kDtypIdx));
    return ss.str();
  }
};

enum class ProdDimIntIndex { kDimIdx = 1, kKeepdimIdx = 2, kDtypeIdx = 3 };
class ProdDimInt : public Node {
 public:
  ProdDimInt() = delete;
  ProdDimInt(
      const at::Tensor& self,
      int64_t dim,
      bool keepdim,
      c10::optional<at::ScalarType> dtype)
      : Node(c10::Symbol::fromQualString("hpu::prod_dim_Int")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);
    auto ir_value = hl_self.GetIrValue();
    AddInput(ir_value);

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, static_cast<size_t>(ProdDimIntIndex::kDimIdx));
    m_meta_data.set(keepdim, static_cast<size_t>(ProdDimIntIndex::kKeepdimIdx));
    m_meta_data.set(dtype, static_cast<size_t>(ProdDimIntIndex::kDtypeIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim="
       << m_meta_data.get(static_cast<size_t>(ProdDimIntIndex::kDimIdx))
       << ", keepdim="
       << m_meta_data.get(static_cast<size_t>(ProdDimIntIndex::kKeepdimIdx))
       << ", dtype="
       << m_meta_data.get(static_cast<size_t>(ProdDimIntIndex::kDtypeIdx));
    return ss.str();
  }
};

enum class AllDimIndex { kDimIdx = 1, kKeepdimIdx = 2 };
class AllDim : public Node {
 public:
  AllDim() = delete;
  AllDim(const at::Tensor& self, int64_t dim, bool keepdim)
      : Node(c10::Symbol::fromQualString("hpu::all_dim")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    hl_self = HbLazyTensorViews::HandleViewsOrUpdate(self, hl_self);
    auto ir_value = hl_self.GetIrValue();
    AddInput(ir_value);

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim, static_cast<size_t>(AllDimIndex::kDimIdx));
    m_meta_data.set(keepdim, static_cast<size_t>(AllDimIndex::kKeepdimIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString()
       << ", dim=" << m_meta_data.get(static_cast<size_t>(AllDimIndex::kDimIdx))
       << ", keepdim="
       << m_meta_data.get(static_cast<size_t>(AllDimIndex::kKeepdimIdx));
    return ss.str();
  }
};
}; // namespace ir
}; // namespace habana_lazy
