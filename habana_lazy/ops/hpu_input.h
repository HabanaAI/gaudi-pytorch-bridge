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

class Input : public Node {
 public:
  Input() = delete;
  Input(const habana_lazy::HbLazyTensor& hl_tensor)
      : Node(c10::Symbol::fromQualString("hpu::input")) {
    HABANA_ASSERT(hl_tensor.is_null() == false);
    m_tensor = habana_lazy::AtenFromHbLazyTensor(hl_tensor);
  }

  /*std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString();
    if (m_tensor.defined()) {
      ss << ", tensor=" << m_tensor.toString();
    }
    return ss.str();
  }*/

  const at::Tensor& GetTensor() const {
    return m_tensor;
  }

 private:
  at::Tensor m_tensor;
};

}; // namespace ir
}; // namespace habana_lazy
