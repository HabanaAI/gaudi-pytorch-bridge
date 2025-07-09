/**
 * Copyright (c) 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <kineto/Config.h>
#include <kineto/libkineto.h>
#pragma GCC diagnostic pop
#include "backend/profiling/profiling.h"

namespace habana::profile {
using namespace std::chrono;

class GenericTraceActivitySink : public TraceSink {
 public:
  explicit GenericTraceActivitySink(
      std::deque<std::unique_ptr<libkineto::GenericTraceActivity>>& activities);
  ~GenericTraceActivitySink() override;

  void addCompleteActivity(
      const Activity& activity,
      const std::optional<RecipeInfo>& recipeInfo,
      uint64_t start,
      uint64_t end) override;

  void finishPendingsActivities(uint64_t time);

  void addActivity(
      const Activity& activity,
      const std::optional<RecipeInfo>& recipeInfo,
      uint64_t time,
      bool begin) override;

  void addMemoryEvent(
      int64_t device,
      int64_t resource,
      int64_t time,
      uint64_t addr,
      int64_t bytes,
      int64_t device_id,
      int64_t device_type,
      uint64_t total_allocated,
      uint64_t total_reserved) override;

  void addDevice(std::string_view name, int64_t device) override;

  void addResource(
      std::string_view name,
      int64_t device,
      int64_t resource,
      int64_t sort_index = -1) override;

  std::string getDeviceDetails();

  void addDeviceDetails(const std::unordered_map<std::string, std::string>&
                            device_properties) override;
  void addDeviceDetails(const std::unordered_map<std::string, int64_t>&
                            device_properties) override;

  std::unique_ptr<libkineto::GenericTraceActivity> constructFlow(
      const std::string& name,
      libkineto::ActivityType type,
      int64_t device,
      int64_t resource,
      int64_t time,
      uint64_t flow_id,
      bool start);

  void addFlowEvent(
      std::string_view name,
      std::string_view,
      const Flow& start,
      const Flow& finish) override;

  void clear() override;

  int64_t transToRelativeTime(int64_t time) override;

  void processTrace(
      libkineto::ActivityLogger& logger,
      int64_t beginTime,
      int64_t endTime);

 private:
  struct PendingActivity {
    Activity activity;
    std::optional<RecipeInfo> recipeInfo;
    uint64_t startTime;
  };
  const libkineto::TraceSpan& defaultTraceSpan();
  libkineto::ActivityType mapHabanaTypeToKinetoType(ActivityType type);

  std::deque<std::unique_ptr<libkineto::GenericTraceActivity>>& activities_;
  std::vector<libkineto::DeviceInfo> deviceInfos_;
  std::vector<libkineto::ResourceInfo> resourceInfos_;

  std::unordered_map<std::string, std::stack<PendingActivity>>
      pendingActivities_;

  uint64_t profiler_event_index_{0};
  uint64_t flow_id_counter_{0};
  std::ostringstream device_properties_;
};

class HPUActivityProfiler : public libkineto::IActivityProfiler {
 public:
  HPUActivityProfiler() = default;
  HPUActivityProfiler(const HPUActivityProfiler&) = delete;
  HPUActivityProfiler& operator=(const HPUActivityProfiler&) = delete;

  const std::string& name() const override;
  const std::set<libkineto::ActivityType>& availableActivities() const override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override;

 private:
  std::string name_{"HPU"};

  const std::set<libkineto::ActivityType> supported_activities{
      libkineto::ActivityType::HPU_OP,
      libkineto::ActivityType::CONCURRENT_KERNEL};
};

class Config {
 public:
  static Config& getInstance();

  void setBridgeProfile(bool value);
  bool isBridgeProfileEnabled();

  Config(const Config&) = delete;
  Config& operator=(const Config&) = delete;

 private:
  bool isBridgeProfile = false;
  std::mutex mutex_;

  Config() = default;
};

class HpuActivityProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  HpuActivityProfilerSession();
  HpuActivityProfilerSession(const HpuActivityProfilerSession&) = delete;
  HpuActivityProfilerSession& operator=(const HpuActivityProfilerSession&) =
      delete;

  void start() override;
  void stop() override;
  std::vector<std::string> errors() override {
    return errors_;
  };
  void processTrace(libkineto::ActivityLogger& logger) override;
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

 private:
  void hideEventIfNeeded(
      std::unique_ptr<libkineto::GenericTraceActivity>& activity);
  bool isMemoryProfileEnabled();
  std::deque<std::unique_ptr<libkineto::GenericTraceActivity>> activities_;
  std::unique_ptr<GenericTraceActivitySink> sink_;
  std::unique_ptr<Profiler> profiler_;
  int64_t profilerStartTs_{0};
  int64_t profilerEndTs_{0};
  std::vector<std::string> errors_ = {};
};

}; // namespace habana::profile
