#pragma once

#include <unordered_map>
#include "pytorch_helpers/habana_helpers/profiling/profiling.h"
#include "pytorch_helpers/habana_helpers/profiling/trace_sources/trace_parser.h"

namespace habana {

class SynapseProfilerSource : public TraceSource {
 public:
  SynapseProfilerSource();
  ~SynapseProfilerSource() = default;

  void start() override;
  void stop() override;
  void extract(TraceSink& output) override;

 private:
  void convertLogs(TraceSink& output);
  void getLogsSize(size_t& size, size_t& count);
  bool getEntries(size_t& size, size_t& count, void* out);
  void initHpuDetails(TraceSink& output);

  std::unique_ptr<HpuTraceParser> parser_;
  long double wall_stop_time_;
};
} // namespace habana