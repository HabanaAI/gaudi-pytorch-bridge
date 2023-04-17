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

#include "backend/helpers/dynamic_shape_info.h"

#include <iostream>

#include "backend/lazy_to_backend.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy/lazy_executor.h"

namespace habana_helpers {
thread_local bool m_enable_refine_dynamic_shape{
    GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)};

void SetRefineDynamicShape(bool flag) {
  m_enable_refine_dynamic_shape = flag;
  if (lazy_to_backend::is_lazy_inference_call_context()) {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES, flag, 1);
  }
}

void EnableRefineDynamicShape() {
  return SetRefineDynamicShape(true);
}

void DisableRefineDynamicShape() {
  return SetRefineDynamicShape(false);
}

bool GetRefineDynamicShapeStatus() {
  // Dynamic shape supported only for lazy
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 1) {
    return false;
  }

  if (!lazy_to_backend::is_lazy_inference_call_context()) {
    return m_enable_refine_dynamic_shape;
  }
  return GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
}

} // namespace habana_helpers
