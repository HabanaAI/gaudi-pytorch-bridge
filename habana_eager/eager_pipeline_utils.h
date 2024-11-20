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

#include <ATen/core/TensorBody.h>
#include "backend/backend_meta.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "eager_tensor.h"
#include "habana_eager/eager_context.h"

namespace habana {
namespace eager {

template <class F, class... Args>
void op_Lowering_Task(F&& f, Args&&... args) {
  f(std::forward<Args>(args)...);
}

// Generic function to move set calls in StorageExtraMeta to
// the lowering thread in order to avoid race conditions.
template <class F, class... Args>
void pipeline_or_direct_generic(F&& f, Args&&... args) {
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE)) {
    habana::eager::ScheduleWorkAndUpdateLoweringThreadHandle(
        op_Lowering_Task<F, Args...>,
        std::forward<F>(f),
        std::forward<Args>(args)...);
  } else {
    std::forward<F>(f)(std::forward<Args>(args)...);
  }
}

} // namespace eager
} // namespace habana
