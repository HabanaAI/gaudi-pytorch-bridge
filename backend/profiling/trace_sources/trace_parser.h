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

#pragma once

#include <strings.h>
#include <memory>
#include "backend/profiling/profiling.h"
#include "synapse_api.h"

namespace habana {
namespace profile {

struct EngineDatabase;

class HpuTraceParser {
 public:
  HpuTraceParser(
      long double hpu_start_time,
      long double wall_start_time,
      unsigned offset_);

  ~HpuTraceParser();

  void Export(
      synTraceEvent* events_ptr,
      size_t num_events,
      long double wall_stop_time,
      TraceSink& trace_sink);

 private:
  bool skipEvent(const synTraceEvent* events_ptr);
  void initLanes(TraceSink& trace_sink);
  bool isEventInTime(
      long double start,
      long double end,
      long double wall_stop_time);
  void processActivity(
      long double event_start_time,
      long double event_end_time,
      long double wall_stop_time,
      synTraceEvent* events_ptr,
      synTraceEvent* enqueue_events_ptr,
      TraceSink& trace_sink);
  void convertEventsToActivities(
      synTraceEvent* events_ptr,
      size_t num_events,
      long double wall_stop_time,
      TraceSink& trace_sink);
  int64_t timeStampHpuToTB(long double t);
  int64_t getDevice(const synTraceEvent* events_ptr);
  bool isEventKernel(const synTraceEvent* events_ptr);
  ActivityType getActivityType(const synTraceEvent* events_ptr);
  std::unordered_map<std::string, std::string> getExtraArgs(
      const synTraceEvent* events_ptr);
  const std::string plane_name_ = "/device:HPU:0";
  long double hpu_start_time_;
  long double wall_start_time_;
  pid_t device_lane_{0};
  std::unique_ptr<EngineDatabase> engine_type_database_;
  unsigned offset_{};
};
}; // namespace profile
}; // namespace habana