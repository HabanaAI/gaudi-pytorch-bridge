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

/**
 * Handle Scalar, Int, Double, Bool type values
 *
 * Creates a prim::Constant node and connects its output
 * to value struct that contains value for Scalar or Double
 * or Int or Bool type
 */
template <typename T>
class Constant : public Node {
 public:
  // For value = None
  Constant() : Node(c10::Symbol::fromQualString("prim::constant")) {
    m_meta_data.set(at::IValue(), 0);
  }

  Constant(T s) : Node(c10::Symbol::fromQualString("prim::constant")) {
    m_meta_data.set(s, 0);
  }

  const torch::jit::IValue getIValue() const {
    return m_meta_data.get(0);
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", value=" << m_meta_data.get(0);
    return ss.str();
  }
};

using ScalarConstant = Constant<c10::Scalar>;

class ListConstruct : public Node {
  bool m_is_optional;

 public:
  ListConstruct() = delete;
  ListConstruct(const ir::ValueList& values, bool optional)
      : Node(c10::Symbol::fromQualString("prim::ListConstruct")),
        m_is_optional{optional} {
    for (auto& v : values) {
      AddInput(v);
    }
  }

  bool isOptional() const {
    return m_is_optional;
  }
};

class OnesLike : public Node {
 public:
  enum class OnesLikeParam {
    kDtypeIdx = 1,
    kLayoutIdx = 2,
    kDeviceIdx = 3,
    kPinMemoryIdx = 4,
    kMemoryFormatIdx = 5
  };

  OnesLike() = delete;

  OnesLike(
      const at::Tensor& self,
      const at::TensorOptions& options,
      c10::optional<c10::MemoryFormat> optional_memory_format)
      : Node(c10::Symbol::fromQualString("aten::ones_like")) {
    auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
    AddInput(hl_self.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{self};
    AddInputPtTensors(input_pt_vec);

    if (options.has_dtype()) {
      m_meta_data.set(
          at::typeMetaToScalarType(options.dtype()),
          static_cast<size_t>(OnesLikeParam::kDtypeIdx));
    } else {
      m_meta_data.set(
          c10::nullopt, static_cast<size_t>(OnesLikeParam::kDtypeIdx));
    }
    m_meta_data.set(
        options.layout_opt(), static_cast<size_t>(OnesLikeParam::kLayoutIdx));
    m_meta_data.set(
        options.device_opt(), static_cast<size_t>(OnesLikeParam::kDeviceIdx));
    m_meta_data.set(
        options.pinned_memory_opt(),
        static_cast<size_t>(OnesLikeParam::kPinMemoryIdx));
    if (optional_memory_format.has_value()) {
      m_meta_data.set(
          optional_memory_format.value(),
          static_cast<size_t>(OnesLikeParam::kMemoryFormatIdx));
    } else {
      m_meta_data.set(
          c10::nullopt, static_cast<size_t>(OnesLikeParam::kMemoryFormatIdx));
    }
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", dtype="
       << m_meta_data.get(static_cast<size_t>(OnesLikeParam::kDtypeIdx))
       << ", layout="
       << m_meta_data.get(static_cast<size_t>(OnesLikeParam::kLayoutIdx))
       << ", device="
       << m_meta_data.get(static_cast<size_t>(OnesLikeParam::kDeviceIdx))
       << ", pin_memory="
       << m_meta_data.get(static_cast<size_t>(OnesLikeParam::kPinMemoryIdx))
       << ", memory_format="
       << m_meta_data.get(static_cast<size_t>(OnesLikeParam::kMemoryFormatIdx));
    return ss.str();
  }
};

} // namespace ir
} // namespace habana_lazy
