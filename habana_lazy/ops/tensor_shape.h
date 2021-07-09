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

class Permute : public ir::Node {
 public:
  enum class PermuteIdx { kDimIdx = 1 };
  Permute() = delete;
  Permute(
      const at::Tensor& self,
      at::IntArrayRef dims,
      std::string op = "aten::permute")
      : Node(c10::Symbol::fromQualString(op)) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dims, static_cast<size_t>(PermuteIdx::kDimIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dims="
       << m_meta_data.get(static_cast<size_t>(PermuteIdx::kDimIdx));
    return ss.str();
  }
};

class Expand : public ir::Node {
 public:
  enum class ExpandIdx { kDimIdx = 1, kImplicitIdx = 2 };
  Expand() = delete;
  Expand(const at::Tensor& self, at::IntArrayRef dims, bool implicit)
      : Node(c10::Symbol::fromQualString("aten::expand")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dims, static_cast<size_t>(ExpandIdx::kDimIdx));
    m_meta_data.set(implicit, static_cast<size_t>(ExpandIdx::kImplicitIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString()
       << ", dims=" << m_meta_data.get(static_cast<size_t>(ExpandIdx::kDimIdx))
       << ", implicit="
       << m_meta_data.get(static_cast<size_t>(ExpandIdx::kDimIdx));
    return ss.str();
  }
};

class Transpose : public ir::Node {
 public:
  enum class TransposeIdx { kDim0Idx = 1, kDim1Idx = 2 };
  Transpose() = delete;
  Transpose(const at::Tensor& self, int64_t dim0, int64_t dim1)
      : Node(c10::Symbol::fromQualString("aten::transpose")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dim0, static_cast<size_t>(TransposeIdx::kDim0Idx));
    m_meta_data.set(dim1, static_cast<size_t>(TransposeIdx::kDim1Idx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dim0="
       << m_meta_data.get(static_cast<size_t>(TransposeIdx::kDim0Idx))
       << ", dim1="
       << m_meta_data.get(static_cast<size_t>(TransposeIdx::kDim1Idx));
    return ss.str();
  }
};

class PermuteCL : public ir::Node {
 public:
  enum class PermuteIdx { kDimIdx = 1 };
  PermuteCL() = delete;
  PermuteCL(const at::Tensor& self, at::IntArrayRef dims)
      : Node(c10::Symbol::fromQualString("hpu::permute_cl")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(dims, static_cast<size_t>(PermuteIdx::kDimIdx));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dims="
       << m_meta_data.get(static_cast<size_t>(PermuteIdx::kDimIdx));
    return ss.str();
  }
};

}; // namespace ir
}; // namespace habana_lazy
