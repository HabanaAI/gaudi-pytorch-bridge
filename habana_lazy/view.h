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
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "habana_helpers/logging.h"
#include "ir.h"

namespace habana_lazy {
namespace ir {
// This class holds the original IR and the at::tensor representing the original
// data. We use this info to add a dependency node whenever a view gets updated
class LazyView {
 public:
  LazyView(at::Tensor at_tensor, ir::Value&& ir_val) {
    at_tensor_ = at_tensor;
    ir_value_ = std::move(ir_val);
  };
  LazyView(){};
  at::Tensor& getAtTensor() {
    return at_tensor_;
  };
  Value& getIR() {
    return ir_value_;
  }
  void updateView() {
    updated = true;
  }
  bool updateStatus() {
    return updated;
  }

 private:
  at::Tensor at_tensor_;
  Value ir_value_;
  bool updated = false;
};
} // namespace ir
} // namespace habana_lazy