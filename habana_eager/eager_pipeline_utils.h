/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include <ATen/core/TensorBody.h>
#include "backend/backend_meta.h"
#include "backend/helpers/eager_pipeline.h"
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
    habana::eager::SingleTonEagerContext::getInstance()
        .ScheduleWorkAndUpdateLoweringThreadHandle(
            op_Lowering_Task<F, Args...>,
            std::forward<F>(f),
            std::forward<Args>(args)...);
  } else {
    std::forward<F>(f)(std::forward<Args>(args)...);
  }
}

} // namespace eager
} // namespace habana
