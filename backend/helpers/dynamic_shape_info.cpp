/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
  auto hpu_mod = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  if (hpu_mod == 1) {
    m_enable_refine_dynamic_shape = flag;
    if (lazy_to_backend::is_lazy_inference_call_context()) {
      SET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES, flag, 1);
    }
  } else if (hpu_mod == 0) {
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
  auto hpu_mod = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  if (hpu_mod == 1) {
    if (!lazy_to_backend::is_lazy_inference_call_context()) {
      return m_enable_refine_dynamic_shape;
    }
    return GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  } else if (hpu_mod == 0) {
    return GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES);
  } else {
    return false;
  }
}

// PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR is used to enable host tensor
// path for arange DS
bool GetArangeHostTensorStatus() {
  return GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR);
}

void SetHybridSIFTorchCompile(bool flag) {
  // [TODO] Disable hybrid sif until SW-153320
  SET_ENV_FLAG_NEW(PT_HPU_RUN_HYBRID_SIF, flag, 1);
  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE, flag, 1);
}

void EnableOpimDynamicOutputSIF() {
  SET_ENV_FLAG_NEW(PT_HPU_OPTIM_DYNAMIC_OUTPUT_SIF, true, 1);
}

void DisableOpimDynamicOutputSIF() {
  SET_ENV_FLAG_NEW(PT_HPU_OPTIM_DYNAMIC_OUTPUT_SIF, false, 1);
}

} // namespace habana_helpers
