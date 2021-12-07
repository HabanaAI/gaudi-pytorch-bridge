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

#include <tuple>
#include <utility>

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/lazy_executor.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

namespace habana_lazy {
bool is_aliased_view(HbLazyTensorImpl& self, HbLazyTensorImpl& other);
} // namespace habana_lazy
