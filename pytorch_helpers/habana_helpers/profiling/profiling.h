#pragma once

#include <strings.h>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace habana {

enum class ActivityType { KERNEL, RUNTIME, MEMCPY, MEMSET };

struct TraceSource;

struct TraceSink {
  virtual ~TraceSink(){};
  virtual void addCompleteActivity(
      const std::string_view& name,
      const std::string_view& func,
      habana::ActivityType type,
      int64_t device,
      int64_t resource,
      uint64_t start,
      uint64_t end) = 0;

  virtual void addActivity(
      const std::string_view& name,
      const std::string_view& func,
      const std::unordered_map<std::string, std::string>& args,
      habana::ActivityType type,
      int64_t device,
      int64_t resource,
      uint64_t time,
      bool begin) = 0;

  virtual void addDevice(const std::string_view& name, int64_t device) = 0;

  virtual void addResource(
      const std::string_view& name,
      int64_t device,
      int64_t resource,
      int64_t sort_index = -1) = 0;

  virtual void addDeviceDetails(
      const std::unordered_map<std::string, std::string>& device_details) = 0;
};

struct TraceSource {
  virtual ~TraceSource(){};
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void extract(TraceSink& output) = 0;
};

class Profiler {
 public:
  Profiler(TraceSink& sink);
  void start();
  void stop();

 private:
  TraceSink& trace_sink_;
  std::vector<std::unique_ptr<TraceSource>> trace_sources_;
};
}; // namespace habana