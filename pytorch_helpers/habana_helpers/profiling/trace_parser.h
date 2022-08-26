#pragma once

#include <strings.h>
#include <deque>
#include <memory>
#include "synapse_api.h"

namespace habana {
struct EngineDatabase;

struct TraceOutput {
  virtual ~TraceOutput(){};
  virtual void addActivity(
      const std::string& name,
      bool isTPC,
      int64_t device,
      int64_t resource,
      uint64_t start,
      uint64_t end) = 0;

  virtual void addDevice(const std::string& name, int64_t device) = 0;

  virtual void addResource(
      const std::string& name,
      int64_t device,
      int64_t resource,
      int64_t sort_index = -1) = 0;
};

class HpuTraceParser {
 public:
  HpuTraceParser(
      TraceOutput& trace_output,
      long double hpu_start_time,
      long double wall_start_time);

  ~HpuTraceParser();

  void Export(
      synTraceEvent* events_ptr,
      size_t num_events,
      long double wall_stop_time);

 private:
  bool skipEvent(const synTraceEvent* events_ptr);
  void initLanes();
  bool isEventInTime(
      long double start,
      long double end,
      long double wall_stop_time);
  void convertEventsToActivities(
      synTraceEvent* events_ptr,
      size_t num_events,
      long double wall_stop_time);
  int64_t timeStampHpuToTB(long double t);
  int64_t getDevice(const synTraceEvent* events_ptr);

  TraceOutput& trace_output_;
  const std::string plane_name_ = "/device:HPU:0";
  long double hpu_start_time_;
  long double wall_start_time_;
  pid_t device_lane_{1};
  std::unique_ptr<EngineDatabase> engine_type_database_;
};
}; // namespace habana