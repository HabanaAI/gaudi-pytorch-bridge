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
