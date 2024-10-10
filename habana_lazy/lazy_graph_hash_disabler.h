/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include <cstddef>

#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"

namespace habana_lazy {

class DisableRunningHashUpdates {
 public:
  DisableRunningHashUpdates(bool terminate_on_access = false)
      : terminate_on_access_(terminate_on_access) {
    if (terminate_on_access_) {
      terminate_on_access_cnt++;
    }
    disable_cnt++;
  }
  ~DisableRunningHashUpdates() {
    if (terminate_on_access_) {
      terminate_on_access_cnt--;
    }
    disable_cnt--;
  }

  static bool IsHashingEnabled() {
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GRAPH_RUNNING_HASH)) {
      return false;
    }
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE) == 0) {
      // TODO: This is only transiently allowed until acc workitems aren't fixed
      // for all ops.
      return true;
    }
    HABANA_ASSERT(!terminate_on_access_cnt);
    return disable_cnt == 0;
  }

 private:
  bool terminate_on_access_;
  static thread_local size_t disable_cnt;
  static thread_local size_t terminate_on_access_cnt;
};

} // namespace habana_lazy
