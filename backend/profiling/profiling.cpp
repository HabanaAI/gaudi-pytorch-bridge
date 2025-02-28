/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "backend/profiling/profiling.h"
#include <sys/types.h>
#include <stdexcept>
#include "backend/profiling/trace_sources/bridge_logs_source.h"
#include "backend/profiling/trace_sources/memory_source.h"
#include "backend/profiling/trace_sources/synapse_profiler_source.h"
#include "backend/synapse_helpers/env_flags.h"

namespace {
constexpr size_t kMaxThreadName = 32;
} // namespace

namespace habana {
namespace profile {

std::string getThreadName() {
  std::array<char, kMaxThreadName + 1> name{};
  int result = pthread_getname_np(pthread_self(), name.data(), name.size());

  if (result != 0 || name[0] == '\0') {
    return "UnnamedThread";
  } else {
    name[kMaxThreadName] = '\0';
    return std::string(name.data());
  }
}

int64_t getOffset(TraceSourceVariant variant) {
  switch (variant) {
    case TraceSourceVariant::SYNAPSE_PROFILER:
      return 0;
    case TraceSourceVariant::BRIDGE_LOGS:
      return 0;
    case TraceSourceVariant::MEMORY_LOGS:
      return 30000;
  }
  return 0;
}

Profiler::Profiler(TraceSink& sink) : trace_sink_{sink} {}

void Profiler::init_sources(
    bool bridge,
    bool memory,
    const std::vector<std::string>& mandatory_events) {
  // if an object contained this class is static,
  // this function called several time in the same object.
  // Need to avoid logger duplication in the list.
  trace_sources_.clear();

  trace_sources_.push_back(std::make_unique<SynapseProfilerSource>());
  if (bridge || !mandatory_events.empty()) {
    trace_sources_.push_back(
        std::make_unique<BridgeLogsSource>(bridge, mandatory_events));
  }
  if (memory) {
    trace_sources_.push_back(std::make_unique<MemorySource>());
  }
  // simple trace grouping by log category
  for (auto& trace_source : trace_sources_) {
    trace_source->set_offset(getOffset(trace_source->get_variant()));
  }
}

void Profiler::start() {
  for (auto& trace_source : trace_sources_) {
    trace_source->start(trace_sink_);
  }
}

void Profiler::stop() {
  for (auto& trace_source : trace_sources_) {
    trace_source->stop();
  }

  for (auto& trace_source : trace_sources_) {
    trace_source->extract(trace_sink_);
  }
}
} // namespace profile
} // namespace habana
