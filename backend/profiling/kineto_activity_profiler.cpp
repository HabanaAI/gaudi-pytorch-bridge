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

#include "kineto_activity_profiler.h"
#include <fmt/format.h>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>
#include "backend/profiling/profiling.h"
#include "backend/profiling/trace_sources/bridge_logs_source.h"
#include "backend/synapse_helpers/env_flags.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <kineto/output_base.h>
#include <kineto/time_since_epoch.h>
#pragma GCC diagnostic pop
#include <common/warning_suppress.h>
#include <torch/csrc/profiler/orchestration/observer.h>
#include <stack>

namespace {
std::string toHex(uint64_t handle) {
  std::stringstream stream;
  stream << "0x" << std::hex << handle;
  return stream.str();
}
std::string toString(std::string_view str) {
  return std::string("\"") + std::string(str) + std::string("\"");
}

//------------------------------------------------------------------------------
// Link-specification table used by GenericTraceActivitySink to create Perfetto
// flow edges between pairs of activities.
//
// Each LinkSpec instance encodes:
//
//   canonical - The canonical (current) activity name.
//   policy    - Indicates where the flow edge starts: at the end or
//                start timestamp of the previous matching activity.
//   prev      - The activity name before the current event.
//   needRemove - If true, remove the previous event..
//
// The table defines when and how two events should be linked together.
//------------------------------------------------------------------------------

bool startsWith(std::string_view s, std::string_view p) noexcept {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

const habana::profile::LinkSpec* findSpec(std::string_view name) noexcept {
  static const std::array<habana::profile::LinkSpec, 4> kLinkSpecs{
      {{"run", habana::profile::TimePolicy::kPrevEnd, "run", true},
       {"Launch", habana::profile::TimePolicy::kPrevEnd, "run", false},
       {"compileGraph", habana::profile::TimePolicy::kPrevStart, "run", true},
       {"enqueueWithExternalEventsExt",
        habana::profile::TimePolicy::kPrevStart,
        "Launch",
        true}}};
  for (const auto& spec : kLinkSpecs)
    if (startsWith(name, spec.canonical))
      return &spec;
  return nullptr;
}

std::string makeKey(
    const std::string& key,
    const habana::profile::Activity& act,
    const std::optional<habana::profile::RecipeInfo>& recipe) {
  std::string k{key};
  k += ":";
  if (auto it = act.args.find("index"); it != act.args.end()) {
    k += it->second;
  } else if (
      recipe && !recipe->recipeName.empty() &&
      habana::profile::RecipeRegistry::hasRecipeName(recipe->recipeName)) {
    auto id = habana::profile::RecipeRegistry::getRecipeId(recipe->recipeName);
    k += std::to_string(id);
  }
  return k;
}

bool shouldHideEvent(
    std::unique_ptr<libkineto::GenericTraceActivity>& activity,
    int64_t startTime,
    int64_t endTime) {
  if (activity->type() != libkineto::ActivityType::PRIVATEUSE1_RUNTIME) {
    return true;
  }

  if (activity->startTime < startTime) {
    return true;
  }

  if (activity->endTime > endTime) {
    return true;
  }

  return false;
}
} // namespace

namespace habana::profile {

using namespace libkineto;
using namespace std::chrono;
GenericTraceActivitySink::GenericTraceActivitySink(
    std::deque<std::unique_ptr<GenericTraceActivity>>& activities)
    : activities_{activities} {}
GenericTraceActivitySink::~GenericTraceActivitySink() = default;

void GenericTraceActivitySink::addCompleteActivity(
    const Activity& activity,
    const std::optional<RecipeInfo>& recipeInfo,
    uint64_t start,
    uint64_t end) {
  if (habana::profile::bridge::linked_events_enabled()) {
    AddLinkedEvent(activity, recipeInfo, start, end);
  }
  auto ev = std::make_unique<GenericTraceActivity>(
      defaultTraceSpan(),
      mapHabanaTypeToKinetoType(activity.type),
      static_cast<std::string>(activity.name));
  ev->startTime = start;
  ev->endTime = end;
  ev->device = static_cast<int32_t>(activity.device);
  ev->resource = static_cast<int32_t>(activity.resource);
  if (recipeInfo) {
    ev->addMetadata("recipeId", recipeInfo->recipeId);
    ev->addMetadata("recipeName", toString(recipeInfo->recipeName));
    ev->addMetadata("streamHandle", toString(toHex(recipeInfo->streamHandle)));
    ev->addMetadata("eventHandle", toString(toHex(recipeInfo->eventHandle)));
  }
  if (activity.type == ActivityType::KERNEL) {
    ev->addMetadata("device", activity.device);
  }

  if (!activity.args.empty()) {
    for (const auto& kv : activity.args) {
      std::string value = toString(kv.second);
      ev->addMetadata(kv.first, value);
    }
  }

  activities_.push_back(std::move(ev));
}

void GenericTraceActivitySink::AddLinkedEvent(
    const Activity& activity,
    const std::optional<RecipeInfo>& recipeInfo,
    uint64_t start,
    uint64_t end) {
  auto pr = popLinkedEvent(activity, recipeInfo, start, end);
  pushLinkedEvent(activity, recipeInfo, start, end);
  if (pr.has_value()) {
    auto [begin, finish] = pr.value();
    addFlowEvent(activity.name, activity.name, begin, finish);
  }
}

void GenericTraceActivitySink::finishPendingsActivities(uint64_t time) {
  for (auto& [key, activityStack] : pendingActivities_) {
    while (!activityStack.empty()) {
      const PendingActivity& pendingActivity = activityStack.top();
      addCompleteActivity(
          pendingActivity.activity,
          pendingActivity.recipeInfo,
          pendingActivity.startTime,
          time);
      activityStack.pop();
    }
  }
  pendingActivities_.clear();
}

void GenericTraceActivitySink::pushLinkedEvent(
    const Activity& activity,
    const std::optional<RecipeInfo>& recipeInfo,
    uint64_t start,
    uint64_t end) {
  if (const auto* spec = findSpec(activity.name)) {
    linked_.try_emplace(
        makeKey(spec->canonical, activity, recipeInfo),
        activity.device,
        activity.resource,
        start,
        end);
  }
}

std::optional<std::pair<Flow, Flow>> GenericTraceActivitySink::popLinkedEvent(
    const Activity& activity,
    const std::optional<RecipeInfo>& recipeInfo,
    uint64_t start,
    uint64_t /*end*/) {
  const auto* spec = findSpec(activity.name);
  if (!spec)
    return std::nullopt;

  auto key = makeKey(spec->prev, activity, recipeInfo);
  auto it = linked_.find(key);
  if (it == linked_.end())
    return std::nullopt;

  const auto& [linkedDevice, linkedResource, prevStart, prevEnd] = it->second;
  uint64_t beginTime =
      (spec->policy == TimePolicy::kPrevEnd) ? prevEnd : prevStart;
  uint64_t finishTime = start;

  Flow begin{linkedDevice, linkedResource, static_cast<int64_t>(beginTime)};
  Flow finish{
      activity.device, activity.resource, static_cast<int64_t>(finishTime)};
  if (spec->needRemove) {
    linked_.erase(it);
  }
  return std::pair{begin, finish};
}

void GenericTraceActivitySink::addActivity(
    const Activity& activity,
    const std::optional<RecipeInfo>& recipeInfo,
    uint64_t time,
    bool begin) {
  std::string key = std::string(activity.name) +
      std::to_string(activity.device) + std::to_string(activity.resource);

  if (begin) {
    pendingActivities_[key].push({activity, recipeInfo, time});
  } else {
    auto it = pendingActivities_.find(key);
    if (it != pendingActivities_.end() && !it->second.empty()) {
      uint64_t start = it->second.top().startTime;
      addCompleteActivity(it->second.top().activity, recipeInfo, start, time);
      it->second.pop();
      if (it->second.empty()) {
        pendingActivities_.erase(it);
      }
    }
  }
}

void GenericTraceActivitySink::addMemoryEvent(
    int64_t device,
    int64_t resource,
    int64_t time,
    uint64_t addr,
    int64_t bytes,
    int64_t device_id,
    int64_t device_type,
    uint64_t total_allocated,
    uint64_t total_reserved) {
  auto ev = std::make_unique<GenericTraceActivity>(
      defaultTraceSpan(),
      libkineto::ActivityType::CPU_INSTANT_EVENT,
      "[memory]");
  ev->device = static_cast<int32_t>(device);
  ev->resource = static_cast<int32_t>(resource);
  ev->startTime = time;
  profiler_event_index_++;
  ev->addMetadata("Addr", addr);
  ev->addMetadata("Bytes", bytes);
  ev->addMetadata("Device Id", device_id);
  ev->addMetadata("Device Type", device_type);
  ev->addMetadata("Profiler Event Index", profiler_event_index_);
  ev->addMetadata("Total Allocated", total_allocated);
  ev->addMetadata("Total Reserved", total_reserved);
  activities_.push_back(std::move(ev));
}

void GenericTraceActivitySink::addDevice(
    std::string_view name,
    int64_t device) {
  int64_t sort_index = device < 8 ? device + 0x1000000LL : device;
  std::string dev_name = static_cast<std::string>(name);
  deviceInfos_.emplace_back(device, sort_index, dev_name, dev_name);
}

void GenericTraceActivitySink::addResource(
    std::string_view name,
    int64_t device,
    int64_t resource,
    int64_t sort_index) {
  resourceInfos_.emplace_back(
      device, resource, sort_index, static_cast<std::string>(name));
}

std::string GenericTraceActivitySink::getDeviceDetails() {
  std::ostringstream oss;
  oss << fmt::format(R"JSON({{ {} }})JSON", device_properties_.str());
  return oss.str();
}

void GenericTraceActivitySink::addDeviceDetails(
    const std::unordered_map<std::string, std::string>& device_properties) {
  for (const auto& pair : device_properties) {
    if (device_properties_.tellp() != 0) {
      device_properties_ << ", ";
    }
    device_properties_ << fmt::format(
        R"JSON(
          "{}": "{}")JSON",
        pair.first,
        pair.second);
  }
}

void GenericTraceActivitySink::addDeviceDetails(
    const std::unordered_map<std::string, int64_t>& device_properties) {
  for (const auto& pair : device_properties) {
    if (device_properties_.tellp() != 0) {
      device_properties_ << ", ";
    }
    device_properties_ << fmt::format(
        R"JSON(
          "{}": {})JSON",
        pair.first,
        pair.second);
  }
}

std::unique_ptr<GenericTraceActivity> GenericTraceActivitySink::constructFlow(
    const std::string& name,
    libkineto::ActivityType type,
    int64_t device,
    int64_t resource,
    int64_t time,
    uint64_t flow_id,
    bool start) {
  auto flow =
      std::make_unique<GenericTraceActivity>(defaultTraceSpan(), type, name);
  flow->device = static_cast<int32_t>(device);
  flow->resource = static_cast<int32_t>(resource);
  flow->startTime = time;
  SUPPRESS_WCONVERSION(flow->flow.id = static_cast<uint32_t>(flow_id);)
  flow->flow.type = kLinkAsyncCpuGpu;
  flow->flow.start = start;
  return flow;
}

void GenericTraceActivitySink::addFlowEvent(
    std::string_view name,
    std::string_view /*cat*/,
    const Flow& startFlow,
    const Flow& finishFlow) {
  if (habana::profile::bridge::linked_events_enabled() or
      GET_ENV_FLAG_NEW(PT_TB_ENABLE_FLOW_EVENTS)) {
    flow_id_counter_++;
    std::string flow_name = std::string(name);
    auto flow_start = constructFlow(
        flow_name,
        libkineto::ActivityType::HPU_OP,
        startFlow.device,
        startFlow.resource,
        startFlow.time,
        flow_id_counter_,
        true);
    auto flow_finish = constructFlow(
        flow_name,
        libkineto::ActivityType::HPU_OP,
        finishFlow.device,
        finishFlow.resource,
        finishFlow.time,
        flow_id_counter_,
        false);
    activities_.push_back(std::move(flow_start));
    activities_.push_back(std::move(flow_finish));
  }
}

void GenericTraceActivitySink::clear() {}

int64_t GenericTraceActivitySink::transToRelativeTime(int64_t time) {
  return time;
}

void GenericTraceActivitySink::processTrace(
    ActivityLogger& logger,
    int64_t beginTime,
    int64_t endTime) {
  logger.handleTraceStart({}, getDeviceDetails());

  for (auto& deviceInfo : deviceInfos_) {
    logger.handleDeviceInfo(deviceInfo, beginTime);
  }

  for (auto& recipeInfo : resourceInfos_) {
    logger.handleResourceInfo(recipeInfo, beginTime);
  }

  finishPendingsActivities(endTime);
}

const TraceSpan& GenericTraceActivitySink::defaultTraceSpan() {
  static TraceSpan span(0, 0, "PyTorch Profiler", "");
  return span;
}

libkineto::ActivityType GenericTraceActivitySink::mapHabanaTypeToKinetoType(
    ActivityType type) {
  switch (type) {
    case ActivityType::KERNEL:
      return libkineto::ActivityType::CONCURRENT_KERNEL;
    case ActivityType::HPU_RUNTIME:
      return libkineto::ActivityType::PRIVATEUSE1_RUNTIME;
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

const std::string& HPUActivityProfiler::name() const {
  return name_;
}

void HpuActivityProfilerSession::hideEventIfNeeded(
    std::unique_ptr<libkineto::GenericTraceActivity>& activity) {
  if (shouldHideEvent(activity, profilerStartTs_, profilerEndTs_)) {
    activity->addMetadata("hidden", "1");
  }
}

const std::set<libkineto::ActivityType>& HPUActivityProfiler::
    availableActivities() const {
  return supported_activities;
}

std::unique_ptr<libkineto::IActivityProfilerSession> HPUActivityProfiler::
    configure(
        const std::set<libkineto::ActivityType>& activity_types,
        [[maybe_unused]] const libkineto::Config& config) {
  auto env = std::getenv("HABANA_PROFILE");
  bool hpu_profiling_available =
      (env != nullptr) && (std::string_view{env} != "0");

  bool hpu_profiling_requested =
      activity_types.find(libkineto::ActivityType::HPU_OP) !=
          activity_types.end() and
      GET_ENV_FLAG_NEW(PT_PYTORCH_PROFILER_USE_KINETO);

  if (hpu_profiling_requested) {
    if (hpu_profiling_available) {
      auto session = std::make_unique<HpuActivityProfilerSession>();
      return session;
    }
  }
  return nullptr;
}

std::unique_ptr<libkineto::IActivityProfilerSession> HPUActivityProfiler::
    configure(
        [[maybe_unused]] int64_t ts_ms,
        [[maybe_unused]] int64_t duration_ms,
        const std::set<libkineto::ActivityType>& activity_types,
        const libkineto::Config& config) {
  return configure(activity_types, config);
}

Config& Config::getInstance() {
  static Config instance;
  return instance;
}

void Config::setBridgeProfile(bool value) {
  std::lock_guard<std::mutex> lock(mutex_);
  isBridgeProfile = value;
}

bool HpuActivityProfilerSession::isMemoryProfileEnabled() {
  return torch::profiler::impl::getProfilerConfig().profile_memory;
}

bool Config::isBridgeProfileEnabled() {
  std::lock_guard<std::mutex> lock(mutex_);
  return isBridgeProfile;
}

HpuActivityProfilerSession::HpuActivityProfilerSession() {
  status_ = TraceStatus::READY;
  profilerStartTs_ = 0;
  profilerEndTs_ = 0;
}

void HpuActivityProfilerSession::start() {
  sink_ = std::make_unique<GenericTraceActivitySink>(activities_);
  profiler_ = std::make_unique<Profiler>(*sink_);
  std::vector<std::string> mandatory_events;
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    mandatory_events = {
        "SyncTensorsGraphInternal",
        "ExecuteCachedGraph",
        "LaunchSyncTensorsGraph",
        "hpu_lazy"};
  } else {
    mandatory_events = {
        "LaunchRecipeTask",
        "add_new_recipe",
        "launch_recipe",
        "launch",
        "run",
        "Launch"};
  }

  bool bridge_profile = Config::getInstance().isBridgeProfileEnabled();
  profiler_->init_sources(
      bridge_profile, isMemoryProfileEnabled(), mandatory_events);
  profilerStartTs_ =
      libkineto::timeSinceEpoch(std::chrono::high_resolution_clock::now());
  profiler_->start();
  status_ = TraceStatus::RECORDING;
}

void HpuActivityProfilerSession::stop() {
  profilerEndTs_ =
      libkineto::timeSinceEpoch(std::chrono::high_resolution_clock::now());
  profiler_->stop();
  status_ = TraceStatus::READY;
}

void HpuActivityProfilerSession::processTrace(ActivityLogger& logger) {
  sink_->processTrace(logger, profilerStartTs_, profilerEndTs_);

  for (auto& activity : activities_) {
    hideEventIfNeeded(activity);
    activity->log(logger);
  }
}

std::unique_ptr<CpuTraceBuffer> HpuActivityProfilerSession::getTraceBuffer() {
  auto buf = std::make_unique<CpuTraceBuffer>();
  buf->activities.swap(activities_);
  return buf;
}

std::unique_ptr<DeviceInfo> HpuActivityProfilerSession::getDeviceInfo() {
  return {};
}

std::vector<ResourceInfo> HpuActivityProfilerSession::getResourceInfos() {
  return {};
}

std::unique_ptr<IActivityProfiler> register_activity_profiler() {
  return std::make_unique<HPUActivityProfiler>();
}

auto register_activity_sink_factory = [] {
  if (GET_ENV_FLAG_NEW(PT_PYTORCH_PROFILER_USE_KINETO)) {
    libkineto::api().registerProfilerFactory(register_activity_profiler);
  }
  return 0;
}();
}; // namespace habana::profile
