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

#pragma once
#include "components/executor.h"

namespace habana {
namespace program {

/*
 * Creates executor for given lazy_jit_graph.
 * The program associated with executor is looked up from cache.
 *
 * TODO: is it enough to use cached_graph_key() to identify program?
 */
std::unique_ptr<Executor> CreateExecutor(
    std::size_t graph_hash,
    const std::shared_ptr<LazyJitGraph>& lazy_jit_graph);

} // namespace program
} // namespace habana
