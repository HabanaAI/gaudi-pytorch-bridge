/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#include "habana_lazy/lazy_graph_hash_disabler.h"

namespace habana_lazy {
thread_local size_t DisableRunningHashUpdates::disable_cnt{0};
thread_local size_t DisableRunningHashUpdates::terminate_on_access_cnt{0};
} // namespace habana_lazy
