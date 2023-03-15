/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "habana_kernels/lazy_kernels.h"
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

} // namespace ir
} // namespace habana_lazy
