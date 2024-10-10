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
#pragma once

#include <cstdint>
#include "backend/synapse_helpers/env_flags.h"

namespace habana_helpers {

extern thread_local bool m_enable_refine_dynamic_shape;

void EnableRefineDynamicShape();

void DisableRefineDynamicShape();

void SetRefineDynamicShape(bool flag);

bool GetRefineDynamicShapeStatus();

bool GetArangeHostTensorStatus();

void SetHybridSIFTorchCompile(bool flag);

void EnableOpimDynamicOutputSIF();

void DisableOpimDynamicOutputSIF();

} // namespace habana_helpers
