#include "pytorch_helpers/habana_helpers/profiling/trace_sources/synapse_logger_source.h"
#include <exception>
#include <iostream>
#include <memory>
#include <string_view>
#include <unordered_set>
#include "absl/strings/str_split.h"

namespace habana {

SynapseLoggerSource::SynapseLoggerSource() {}

void SynapseLoggerSource::start() {}

void SynapseLoggerSource::stop() {}

void SynapseLoggerSource::extract(TraceSink&) {}
} // namespace habana