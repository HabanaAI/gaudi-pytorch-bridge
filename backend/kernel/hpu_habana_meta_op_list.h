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

#include <iostream>
#include <string>
#include <unordered_set>

namespace habana {

class HabanaMetaOpList {
 private:
  static const std::unordered_set<std::string_view>& meta_ops() {
    static const std::unordered_set<std::string_view> meta_ops_list = {
        // Add aten string here for ops to support
        // e.g  :: "aten::view"
        "aten::size",
        "prim::dtype",
    };
    return meta_ops_list;
  }

 public:
  static bool isHabanaMetaOp(const std::string_view op_name) {
    return (meta_ops().find(op_name) != meta_ops().end());
  }
};

} // namespace habana
