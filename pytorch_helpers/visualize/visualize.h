/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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
#include <torch/csrc/jit/ir/ir.h>
#include <string>

namespace visualize {

using GraphPtr = std::shared_ptr<torch::jit::Graph>;

// Gets or creates a hash to index mapping; use hash itself if map too big
size_t GetGraphIndex(size_t hash);

// Dumps a JIT IR before optimizations
void DumpPreGraph(const GraphPtr& graph, size_t hash);

// Dumps a JIT IR after all optimizations
void DumpPostGraph(const GraphPtr& graph, size_t hash);

// Dumps an JIT IR after an optimization pass
void DumpOptimizedGraph(
    const GraphPtr& graph,
    size_t hash,
    const std::string& pass);

// Dumps a cached JIT IR
void DumpCachedGraph(const GraphPtr& graph, size_t hash);

// Generic JIT IR dump function
void DumpGraph(const GraphPtr& graph, const std::string& filename);

// Dump JIT IR eager graph
void DumpEagerOrCompileGraph(
    const GraphPtr& graph,
    const std::string& graph_name);

} // namespace visualize