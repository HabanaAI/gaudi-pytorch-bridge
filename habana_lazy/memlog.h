/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <string_view>

// namespace habana_lazy
namespace habana_lazy {

void log_dev_mem_stats(
    std::string_view msg,
    std::string_view name = "",
    uint64_t size = 0);

const std::pair<uint64_t, uint32_t> get_future_memory();

} // namespace habana_lazy
