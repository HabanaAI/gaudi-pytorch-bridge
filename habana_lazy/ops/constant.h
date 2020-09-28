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
  Constant() = delete;
  Constant(T s)
      : Node(c10::Symbol::fromQualString("prim::constant")),
        m_ival(torch::jit::IValue(s)) {}

  const torch::jit::IValue& getIValue() const {
    return m_ival;
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", value(" << m_ival << ")";
    return ss.str();
  }

 private:
  torch::jit::IValue m_ival;
};

using ScalarConstant = Constant<c10::Scalar>;

} // namespace ir
}; // namespace habana_lazy