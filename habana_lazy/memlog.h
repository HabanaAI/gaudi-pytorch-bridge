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

// namespace habana_lazy
namespace habana_lazy {

void log_dev_mem_stats(
    const std::string& msg,
    const std::string& name = "",
    uint64_t size = 0);

} // namespace habana_lazy
