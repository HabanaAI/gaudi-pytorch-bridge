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

#include <iostream>
#include <string_view>
#include "backend/profiling/profiling.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "Config.h"
#include "libkineto.h"
#pragma GCC diagnostic pop

namespace habana {
namespace profile {

using namespace libkineto;
using namespace std::chrono;

class GenericTraceActivitySink : public TraceSink {
 public:
  GenericTraceActivitySink(std::deque<GenericTraceActivity>& activities)
      : activities_{activities} {}
  ~GenericTraceActivitySink() {}

  void addCompleteActivity(
      const Activity& activity,
      const std::optional<RecipeInfo>&,
      uint64_t start,
      uint64_t end) override {
    GenericTraceActivity ev{
        defaultTraceSpan(),
        mapHabanaTypeToKinetoType(activity.type),
        static_cast<std::string>(activity.name)};
    ev.startTime = start;
    ev.endTime = end;
    ev.device = activity.device;
    ev.resource = activity.resource;
    if (activity.type == ActivityType::KERNEL) {
      ev.addMetadata("device", activity.device);
    }
    activities_.push_back(ev);
  }

  void addActivity(
      const Activity&,
      const std::optional<RecipeInfo>&,
      uint64_t,
      bool) override {}

  virtual void addMemoryEvent(
      int64_t,
      int64_t,
      int64_t,
      uint64_t,
      int64_t,
      int64_t,
      int64_t,
      uint64_t,
      uint64_t) override {}

  void addDevice(std::string_view name, int64_t device) override {
    GenericTraceActivity name_meta{
        defaultTraceSpan(), libkineto::ActivityType::HPU_META_OP, ""};
    name_meta.startTime = 0;
    name_meta.endTime = 0;
    name_meta.activityName = "process_name";
    name_meta.device = device;
    name_meta.resource = 0;
    name_meta.addMetadata(
        "name", std::string("\"") + static_cast<std::string>(name) + "\"");
    activities_.push_back(name_meta);

    GenericTraceActivity sort_meta{
        defaultTraceSpan(), libkineto::ActivityType::HPU_META_OP, ""};
    sort_meta.startTime = 0;
    sort_meta.endTime = 0;
    sort_meta.device = device;
    sort_meta.resource = 0;
    sort_meta.activityName = "process_sort_index";
    sort_meta.addMetadata(
        "sort_index",
        std::to_string(device < 8 ? device + 0x1000000ll : device));
    activities_.push_back(sort_meta);
  }

  void addResource(
      std::string_view name,
      int64_t device,
      int64_t resource,
      int64_t sort_index = -1) override {
    GenericTraceActivity name_meta{
        defaultTraceSpan(), libkineto::ActivityType::HPU_META_OP, ""};
    name_meta.startTime = 0;
    name_meta.endTime = 0;
    name_meta.activityName = "thread_name";
    name_meta.device = device;
    name_meta.resource = resource;
    name_meta.addMetadata(
        "name", std::string("\"") + static_cast<std::string>(name) + "\"");
    activities_.push_back(name_meta);

    GenericTraceActivity sort_meta{
        defaultTraceSpan(), libkineto::ActivityType::HPU_META_OP, ""};
    sort_meta.startTime = 0;
    sort_meta.endTime = 0;
    sort_meta.device = device;
    sort_meta.resource = resource;
    sort_meta.activityName = "thread_sort_index";
    sort_meta.addMetadata("sort_index", std::to_string(sort_index));
    activities_.push_back(sort_meta);
  }

  void addDeviceDetails(
      const std::unordered_map<std::string, std::string>&) override {}

  virtual void addDeviceDetails(
      const std::unordered_map<std::string, int64_t>&) override {}

  virtual void addFlowEvent(
      std::string_view,
      std::string_view,
      const Flow&,
      const Flow&) override {}

  /**
   * @brief Clean-up data in the object.
   *
   * Clean-up and reset data containers and variables in the object.
   */
  virtual void clear() override {}

  virtual int64_t transToRelativeTime(int64_t time) override {
    return time;
  }

 private:
  const TraceSpan& defaultTraceSpan() {
    static TraceSpan span(0, 0, "PyTorch Profiler", "");
    return span;
  }
  libkineto::ActivityType mapHabanaTypeToKinetoType(ActivityType type) {
    switch (type) {
      case ActivityType::KERNEL:
        return libkineto::ActivityType::CONCURRENT_KERNEL;
      case ActivityType::RUNTIME:
        return libkineto::ActivityType::HPU_OP;
      case ActivityType::MEMCPY:
        return libkineto::ActivityType::GPU_MEMCPY;
      case ActivityType::MEMSET:
        return libkineto::ActivityType::GPU_MEMSET;
      default:
        return libkineto::ActivityType::HPU_OP;
    }
  }
  std::deque<GenericTraceActivity>& activities_;
};

class ProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  explicit ProfilerSession(int64_t, int64_t) {
    status_ = TraceStatus::READY;
    sink_ = std::make_unique<GenericTraceActivitySink>(activities_);
    profiler_ = std::make_unique<Profiler>(*sink_);
  }

  void start() override {
    profiler_->start();
    status_ = TraceStatus::RECORDING;
  }

  void stop() override {
    profiler_->stop();
    status_ = TraceStatus::READY;
  }

  std::vector<std::string> errors() override {
    return {};
  }

  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
    auto buf = std::make_unique<libkineto::CpuTraceBuffer>();
    buf->activities.swap(activities_);
    return buf;
  }

  void processTrace(ActivityLogger& logger) override {
    for (const auto& activity : activities_) {
      activity.log(logger);
    }
  }

 private:
  std::deque<GenericTraceActivity> activities_;
  std::unique_ptr<GenericTraceActivitySink> sink_;
  std::unique_ptr<Profiler> profiler_;
};

class ActivityProfiler : public libkineto::IActivityProfiler {
 public:
  ActivityProfiler() {}
  virtual ~ActivityProfiler() override {}

  virtual const std::string& name() const override {
    return device_name;
  }

  virtual const std::set<libkineto::ActivityType>& availableActivities()
      const override {
    return supported_activities;
  }

  virtual std::unique_ptr<IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activity_types,
      const KINETO_NAMESPACE::Config& config) override {
    auto start_time_ms =
        duration_cast<milliseconds>(system_clock::now().time_since_epoch())
            .count();
    return configure(start_time_ms, 0, activity_types, config);
  }

  virtual std::unique_ptr<IActivityProfilerSession> configure(
      int64_t start_time_ms,
      int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const KINETO_NAMESPACE::Config&) override {
    auto env = std::getenv("HABANA_PROFILE");
    bool hpu_profiling_available =
        (env != nullptr) && (std::string_view{env} != "0");
    bool hpu_profiling_requested =
        activity_types.find(libkineto::ActivityType::HPU_OP) !=
            activity_types.end() ||
        activity_types.find(libkineto::ActivityType::HPU_META_OP) !=
            activity_types.end();

    if (hpu_profiling_requested) {
      if (hpu_profiling_available) {
        auto session =
            std::make_unique<ProfilerSession>(start_time_ms, duration_ms);
        return session;
      } else {
        std::cerr << "Tensorboard callback for HPU hardware profiling disabled"
                     ". To enable set \"HABANA_PROFILE\""
                  << std::endl;
      }
    }
    return nullptr;
  }

 private:
  const std::set<libkineto::ActivityType> supported_activities{
      libkineto::ActivityType::HPU_OP,
      libkineto::ActivityType::CONCURRENT_KERNEL,
      libkineto::ActivityType::HPU_META_OP};
  std::string device_name{"HPU"};
};

std::unique_ptr<IActivityProfiler> register_activity_profiler() {
  return std::make_unique<ActivityProfiler>();
}
}; // namespace profile
}; // namespace habana