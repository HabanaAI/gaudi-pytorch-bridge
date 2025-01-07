/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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