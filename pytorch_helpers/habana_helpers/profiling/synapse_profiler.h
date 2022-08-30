#pragma once

#include <unordered_map>
#include "trace_parser.h"

namespace habana {

struct HPUDetailsConsumer {
  virtual void add_device_details(
      const std::unordered_map<std::string, std::string>& device_details) = 0;
};

class SynapseProfiler : public TraceOutput {
 public:
  SynapseProfiler(HPUDetailsConsumer& hpu_details_consumer);
  void start();
  void stop();
  uint64_t startCustomMeasurement(const std::string& tag);
  void stopCustomMeasurement(uint64_t id);

 private:
  void convertLogs();
  void getLogsSize(size_t& size, size_t& count);
  bool getEntries(size_t& size, size_t& count, void* out);
  bool dumpEntries();
  void init_hpu_details(HPUDetailsConsumer& hpu_details_consumer);

  bool dump_hltv_{false};
  std::unique_ptr<HpuTraceParser> parser_;
  long double wall_stop_time_;
  std::unordered_map<uint64_t, std::pair<uint64_t, std::string>>
      custom_measurements_;
  HPUDetailsConsumer& hpu_details_consumer_;
};
} // namespace habana