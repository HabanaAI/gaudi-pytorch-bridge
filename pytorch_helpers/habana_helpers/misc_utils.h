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

#include <cstdlib>

#include <sys/sysinfo.h>
#include <iostream>
#include <string>

#include <ATen/Tensor.h>
#include "backend/synapse_helpers/env_flags.h"
namespace habana {

// Returns if the input is a ZST
inline bool is_ZST(const at::Tensor& t) {
  // NOTE: There are cases where the numel returns 0 but the tensor has non-null
  // storage pointer. REPRODUCER: LazyDynamicShapesTest.ArangeTest
  return (
      (t.has_storage() && t.storage().data_ptr().get() == nullptr) ||
      (t.numel() == 0));
}

// Computes (x^y)%1000000007
inline int64_t mod_exp(int64_t y, int64_t x = 997) {
  const int64_t p{1000000007};
  int64_t z = 1;
  int64_t sign{(y < 0 ? -1 : 1)};
  // llabs function doesn't handle INT64_MIN value correctly, calling it with
  // such value causes an undefined behaviour. So abs value of INT64_MIN should
  // be calculated manually
  uint64_t yAbs = y == INT64_MIN ? static_cast<uint64_t>(INT64_MAX) + 1
                                 : static_cast<uint64_t>(llabs(y));

  x = x % p;
  if (x == 0) {
    return 0;
  }
  while (yAbs > 0) {
    if (yAbs & 1) {
      z = (z * x) % p;
    }

    yAbs >>= 1;
    x = (x * x) % p;
  }
  z *= sign;
  return z;
}

inline int64_t mod_exp(bool w, int64_t x = 997) {
  int64_t y = (w ? 97 : 43);
  return (mod_exp(y, x));
}

bool IsHostMemoryThresholdReached();
int GetRankFromEnv();
void TryJoinPendingEagerPipelineThreads();
void TryRestoreToOrgSendTensors(
    std::vector<at::Tensor>& tensors,
    std::vector<at::Tensor>& org_tensors);

} // namespace habana
