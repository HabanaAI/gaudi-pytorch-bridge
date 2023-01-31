#pragma once

#include <strings.h>
#include <unordered_map>
#include "pytorch_helpers/habana_helpers/profiling/profiling.h"

namespace habana {

class SynapseLoggerSource : public TraceSource {
 public:
  SynapseLoggerSource();
  virtual ~SynapseLoggerSource() = default;
  void start() override;
  void stop() override;
  void extract(TraceSink& trace_sink) override;
};
} // namespace habana