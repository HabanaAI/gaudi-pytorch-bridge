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
#include <c10/core/DeviceGuard.h>
#include <torch_ver/csrc/distributed/c10d/reducer_timer.hpp>
#include "habana_helpers/logging.h"

namespace c10d {
namespace {

class HpuTimer : public Timer {
 public:
  explicit HpuTimer(c10::Device /* unused */) {}

  c10::optional<int64_t> measureDifference(Event start, Event end) override {
    int64_t start_time = getTimeRef(start);
    int64_t end_time = getTimeRef(end);
    PT_DISTRIBUTED_DEBUG("start time::", start_time, " End time::", end_time);
    if (end_time < start_time) {
      return c10::nullopt;
    }
    return end_time - start_time;
  }
};

C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kHPU, HpuTimer);

} // namespace
} // namespace c10d
