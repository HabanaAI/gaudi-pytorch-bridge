/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include <fstream>
#include <iostream>
#include <mutex>

#include "backend/synapse_helpers/env_flags.h"
#include "serialize/export.h"
#include "visualize.h"

namespace visualize {

std::mutex s_mutex;
std::unordered_map<size_t, size_t> s_graphIndexMap;
size_t s_graphIndex = 0;
ssize_t s_passIndex = 0;

// Gets or creates a hash to index mapping; use hash itself if map too big
size_t GetGraphIndex(size_t hash) {
  std::lock_guard<std::mutex> guard(s_mutex);
  size_t graphIndex = hash;
  if (s_graphIndexMap.count(hash) == 0) {
    if (s_graphIndexMap.size() <
        std::numeric_limits<typeof(s_graphIndex)>::max()) {
      s_graphIndexMap[hash] = s_graphIndex;
      graphIndex = s_graphIndex;
      s_graphIndex++;
    }
  } else {
    graphIndex = s_graphIndexMap[hash];
  }

  return graphIndex;
}

ssize_t ResetPassIndex() {
  std::lock_guard<std::mutex> guard(s_mutex);
  s_passIndex = 0;
  return s_passIndex;
}

ssize_t NextPassIndex() {
  std::lock_guard<std::mutex> guard(s_mutex);
  if (s_passIndex == std::numeric_limits<typeof(s_passIndex)>::max()) {
    s_passIndex = 0;
  }
  return ++s_passIndex;
}

std::string GetGraphFilename(
    const std::string& suffix,
    size_t graphIndex,
    ssize_t passIndex = -1) {
  std::string folder = GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
  std::stringstream ss;
  ss << folder << "/"
     << "jit_ir_" << graphIndex << "_";
  if (passIndex >= 0) {
    ss << passIndex << "_";
  }
  ss << suffix << ".pbtxt";
  return ss.str();
}

void DumpGraph(const GraphPtr& graph, const std::string& filename) {
  std::ofstream ostrm(filename, std::ios::trunc);
  ostrm << serialize::GraphToProtoString(graph);
}

void DumpPreGraph(const GraphPtr& graph, size_t hash) {
  if (GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP) >= 1) {
    DumpGraph(
        graph,
        GetGraphFilename("pre-graph", GetGraphIndex(hash), ResetPassIndex()));
  }
}

void DumpPostGraph(const GraphPtr& graph, size_t hash) {
  if (GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP) >= 1) {
    DumpGraph(
        graph,
        GetGraphFilename("post-graph", GetGraphIndex(hash), NextPassIndex()));
  }
}

void DumpOptimizedGraph(
    const GraphPtr& graph,
    size_t hash,
    const std::string& pass) {
  if (GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP) >= 2) {
    DumpGraph(
        graph,
        GetGraphFilename(
            "after-" + pass, GetGraphIndex(hash), NextPassIndex()));
  }
}

void DumpCachedGraph(const GraphPtr& graph, size_t hash) {
  if (GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP) >= 3) {
    DumpGraph(graph, GetGraphFilename("cached", GetGraphIndex(hash)));
  }
}

void DumpEagerOrCompileGraph(
    const GraphPtr& graph,
    const std::string& graph_name) {
  std::stringstream ss;
  std::string folder = GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
  ss << folder << "/" << graph_name << ".pbtxt";
  DumpGraph(graph, ss.str());
}

} // namespace visualize