/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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