/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <ATen/ATen.h>

namespace at {
namespace habana {

at::Allocator* getHABANADeviceAllocator();

} // namespace habana
} // namespace at
