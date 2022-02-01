/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <memory>
#include "hpu_op_helper.h"

namespace habana {

std::shared_ptr<void> FillDivModParams(size_t& size, bool pyCompatible = true);

} // namespace habana