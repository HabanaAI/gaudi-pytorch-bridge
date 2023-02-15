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

Profiler::Profiler(TraceSink& sink) : trace_sink_{sink} {
  trace_sources_.emplace_back(std::make_unique<SynapseProfilerSource>());
  if (GET_ENV_FLAG_NEW(PT_PROFILE_SYNAPSE_LOGS)) {
    trace_sources_.emplace_back(std::make_unique<SynapseLoggerSource>());
  }
  if (GET_ENV_FLAG_NEW(PT_PROFILE_BRIDGE_LOGS)) {
    trace_sources_.emplace_back(std::make_unique<BridgeLogsSource>());
  }
  if (GET_ENV_FLAG_NEW(PT_PROFILE_MEMORY)) {
    trace_sources_.emplace_back(std::move(std::make_unique<MemorySource>()));
  }

  // simple trace grouping by log category
  for (auto& trace_source : trace_sources_) {
    trace_source->set_offset(
        static_cast<unsigned>(trace_source->get_variant()));
  }
}

void Profiler::start() {
  for (auto& trace_source : trace_sources_) {
    trace_source->start();
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