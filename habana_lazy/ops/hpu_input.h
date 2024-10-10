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
      : Node(c10::Symbol::fromQualString("hpu::input"), true) {
    HABANA_ASSERT(hl_tensor.is_null() == false);
  }
};

}; // namespace ir
}; // namespace habana_lazy
