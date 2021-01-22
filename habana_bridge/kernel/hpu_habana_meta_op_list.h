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

class HabanaMetaOpList {
 private:
  const static std::unordered_set<std::string> HabanaMetaOpsList;

 public:
  static bool isHabanaMetaOp(std::string opName);
};

const std::unordered_set<std::string> HabanaMetaOpList::HabanaMetaOpsList = {
    // Add aten string here for ops to support
    // e.g  :: "aten::view"
    "aten::size",
    "prim::dtype"};

bool HabanaMetaOpList::isHabanaMetaOp(std::string opName) {
  if (HabanaMetaOpsList.find(opName) != HabanaMetaOpsList.end())
    return true;
  return false;
}
