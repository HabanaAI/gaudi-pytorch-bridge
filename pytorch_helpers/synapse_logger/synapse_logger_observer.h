#pragma once
#include <sys/types.h>
#include <string_view>

namespace synapse_logger {
class SynapseLoggerObserver {
 public:
  virtual ~SynapseLoggerObserver(){};
  virtual void on_log(
      std::string_view name,
      std::string_view args,
      pid_t pid,
      pid_t tid,
      int64_t dtime,
      bool is_begin) = 0;
  virtual bool enabled(std::string_view name) = 0;
};

extern "C" void register_synapse_logger_oberver(
    SynapseLoggerObserver* synapse_logger_observer);
}; // namespace synapse_logger