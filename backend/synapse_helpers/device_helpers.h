/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include <synapse_common_types.h>

namespace synapse_helpers {

bool device_supports_fp8(synDeviceType device_type);
bool device_supports_fp16(synDeviceType device_type);
bool device_supports_trunc(synDeviceType device_type);

} // namespace synapse_helpers
