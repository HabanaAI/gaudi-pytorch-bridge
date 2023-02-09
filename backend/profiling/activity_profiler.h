/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************
 */

#pragma once
#include <string.h>
#include <string_view>

namespace habana {
namespace profile {
void export_profiler_logs(std::string_view path);
void start_profiler_session();
void stop_profiler_session();
}; // namespace profile
}; // namespace habana