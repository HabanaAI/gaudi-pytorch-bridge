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

class ListUnpack : public Node {
 public:
  ListUnpack() = delete;
  ListUnpack(const ir::Value value)
      : Node(c10::Symbol::fromQualString("prim::ListUnpack")) {
    AddInput(value);
  }
};

} // namespace ir
} // namespace habana_lazy
