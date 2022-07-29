#pragma once

#include <unordered_map>
#include "trace_parser.h"

namespace habana {

class SynapseProfiler : public TraceOutput {
 public:
  SynapseProfiler();
  void start();
  void stop();
  uint64_t startCustomMeasurement(const std::string& tag);
  void stopCustomMeasurement(uint64_t id);

 private:
  void convertLogs();
  void getLogsSize(size_t& size, size_t& count);
  bool getEntries(size_t& size, size_t& count, void* out);
  bool dumpEntries();

  bool dump_hltv_{false};
  std::unique_ptr<HpuTraceParser> parser_;
  std::unordered_map<uint64_t, std::pair<uint64_t, std::string>>
      custom_measurements_;
};
} // namespace habana