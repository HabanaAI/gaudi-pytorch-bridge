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

#include "backend/profiling/profiling.h"
#include <stdexcept>
#include "backend/profiling/trace_sources/bridge_logs_source.h"
#include "backend/profiling/trace_sources/memory_source.h"
#include "backend/profiling/trace_sources/synapse_logger_source.h"
#include "backend/profiling/trace_sources/synapse_profiler_source.h"
#include "backend/synapse_helpers/env_flags.h"

namespace habana {
namespace profile {

Profiler::Profiler(TraceSink& sink) : trace_sink_{sink} {}

void Profiler::init_sources(
    bool synapse_logger,
    bool bridge,
    bool memory,
    const std::vector<std::string>& mandatory_events) {
  // if an object contained this class is static,
  // this function called several time in the same object.
  // Need to avoid logger duplication in the list.
  trace_sources_.clear();

  trace_sources_.push_back(std::make_unique<SynapseProfilerSource>());
  if (synapse_logger || !mandatory_events.empty()) {
    trace_sources_.push_back(std::make_unique<SynapseLoggerSource>(
        synapse_logger, mandatory_events));
  }
  if (bridge || !mandatory_events.empty()) {
    trace_sources_.push_back(
        std::make_unique<BridgeLogsSource>(bridge, mandatory_events));
  }
  if (memory) {
    trace_sources_.push_back(std::make_unique<MemorySource>());
  }
  // simple trace grouping by log category
  for (auto& trace_source : trace_sources_) {
    trace_source->set_offset(
        static_cast<unsigned>(trace_source->get_variant()));
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