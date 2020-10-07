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
#include "habana_lazy/ir.h"

namespace habana_lazy {

/**
 * Handle Scalar, Int, Double, Bool type values
 *
 * Creates a prim::Constant node and connects its output
 * to value struct that contains value for Scalar or Double
 * or Int or Bool type
 */
template <typename T>
class Constant {
 public:
  Constant() = delete;
  Constant(T val) {
    mp_node = std::make_shared<Node>(
        c10::Symbol::fromQualString("prim::constant"), 1);
    mp_ir_value = std::make_shared<Value>(val, 0);
    mp_ir_value->SetNode(mp_node);
  }

  Value IrValue() {
    return *mp_ir_value.get();
  }

 private:
  NodePtr mp_node;
  ValuePtr mp_ir_value;
};

}; // namespace habana_lazy