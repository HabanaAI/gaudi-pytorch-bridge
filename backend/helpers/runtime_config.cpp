/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "backend/helpers/runtime_config.h"
#include "backend/synapse_helpers/env_flags.h"

namespace habana_helpers {
bool enable_inference_mode{GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE)};

void EnableInferenceMode() {
  enable_inference_mode = true;
}

void DisableInferenceMode() {
  enable_inference_mode = false;
}

bool IsInferenceMode() {
  return enable_inference_mode;
}

} // namespace habana_helpers
