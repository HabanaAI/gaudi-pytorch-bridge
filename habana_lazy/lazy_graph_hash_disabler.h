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
