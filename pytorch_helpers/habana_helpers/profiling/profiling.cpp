#include "pytorch_helpers/habana_helpers/profiling/profiling.h"
#include <stdexcept>
#include "pytorch_helpers/habana_helpers/profiling/trace_sources/synapse_logger_source.h"
#include "pytorch_helpers/habana_helpers/profiling/trace_sources/synapse_profiler_source.h"

using namespace habana;

Profiler::Profiler(TraceSink& sink) : trace_sink_{sink} {
  trace_sources_.push_back(
      std::move(std::make_unique<SynapseProfilerSource>()));
  try {
    trace_sources_.push_back(
        std::move(std::make_unique<SynapseLoggerSource>()));
  } catch (std::runtime_error&) {
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