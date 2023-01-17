/*****************************************************************************
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
#include <string>
#include <unordered_set>
#include "habana_helpers/logging.h"

namespace habana {

class HabanaMetaOpList {
 private:
  const static std::unordered_set<std::string> meta_ops;

 public:
  static bool isHabanaMetaOp(std::string op_name) {
    return (meta_ops.find(op_name) != meta_ops.end());
  }
};

} // namespace habana
