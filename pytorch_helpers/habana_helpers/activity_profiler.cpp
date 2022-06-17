#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <list>
#include <memory>
#include "absl/strings/string_view.h"
#include "synapse_api.h"
#define FMT_HEADER_ONLY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "spdlog/common.h"
#include "spdlog/fmt/bundled/format.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "Config.h"
#include "libkineto.h"
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"
#pragma GCC diagnostic pop

namespace habana {

using namespace libkineto;
using namespace std::chrono;

enum EventType { begin = 'B', end = 'E', metadata = 'M', complete = 'X' };

const char* StringOrFallback(const char* main, const char* fallback) {
  return (main == nullptr or std::strlen(main) == 0) ? fallback : main;
}

uint64_t nowNanos(clockid_t clock) {
  constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;
  struct timespec ts;
  clock_gettime(clock, &ts);
  return (
      static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos +
      static_cast<uint64_t>(ts.tv_nsec));
}

struct EngineType {
  struct Engine {
    uint32_t index;
    std::string name;
  };
  std::string name;
  std::vector<Engine> engines;

  static int getIdx(const std::string_view name) {
    static std::vector<std::string> engines_of_interest = {
        "**DMA ",
        "**MME ",
        "**TPC ", // Gaudi1
        "*PDMA",
        "*EDMA ",
        "*KDMA",
        "*MME ",
        "*TPC ", // Gaudi2
        "*PSOC",
        "*SM",
        "*PMMU",
        "*ROTATOR",
        "*ARC_FARM",
        "*VIDEO_DECODER", // Additional engines in Gaudi2
    };
    for (int i = 0; i < (int)engines_of_interest.size(); i++) {
      if (name.find(engines_of_interest[i]) == 0) {
        return i;
      }
    }
    return -1;
  }
  static bool isInteresting(const std::string_view name) {
    return getIdx(name) != -1;
  }
  static bool isHost(const std::string_view name) {
    static std::string host_meta_name = "***Host";
    return name == host_meta_name;
  }
  static bool isTPC(const std::string_view name) {
    static std::vector<std::string> tpc_engines = {"**TPC ", "*TPC "};
    for (int i = 0; i < (int)tpc_engines.size(); i++) {
      if (name.find(tpc_engines[i]) == 0) {
        return true;
      }
    }
    return false;
  }
};

struct EngineDatabase {
  std::unordered_map<uint32_t, EngineType> engine_types_;
  std::unordered_map<uint32_t, uint32_t> line_info_;

  uint32_t getLine(uint32_t index) {
    auto it = line_info_.find(index);
    if (it != line_info_.end()) {
      return it->second;
    }
    return index;
  }

  void setLine(const EngineType& engine_type, uint32_t index) {
    const auto seperator = 1000;
    auto idx = EngineType::getIdx(engine_type.name);
    line_info_[index] = idx * seperator + index;
  }

  bool isEngineTypeHost(uint32_t engine_type) {
    auto engine_type_it = engine_types_.find(engine_type);
    return engine_type_it != engine_types_.end() &&
        EngineType::isHost(engine_type_it->second.name);
  }

  bool isEngineTypeTPC(uint32_t engine_type) {
    auto engine_type_it = engine_types_.find(engine_type);
    return engine_type_it != engine_types_.end() &&
        EngineType::isTPC(engine_type_it->second.name);
  }

  static EngineDatabase buildDatabase(
      synTraceEvent* events_ptr,
      size_t num_events) {
    EngineDatabase result;
    auto& engine_types = result.engine_types_;

    // Create host engine type first
    synTraceEvent* host_meta_event_ptr = events_ptr;
    for (uint64_t i = 0; i < num_events; i++, host_meta_event_ptr++) {
      if (host_meta_event_ptr->type != EventType::metadata)
        break;
      if (host_meta_event_ptr->engineIndex == 0 &&
          EngineType::isHost(host_meta_event_ptr->arguments.name)) {
        auto& engine_type = engine_types[events_ptr->engineType];
        engine_type.name = host_meta_event_ptr->arguments.name;
        break;
      }
    }

    // Create device engine type and populate with engine names
    for (uint64_t i = 0; i < num_events; i++, events_ptr++) {
      if (events_ptr->type != EventType::metadata) {
        break;
      }
      if (events_ptr->engineIndex == 0 &&
          EngineType::isInteresting(events_ptr->arguments.name)) {
        auto& engine_type = engine_types[events_ptr->engineType];
        engine_type.name = events_ptr->arguments.name;
        continue;
      }
      auto engine_type_it = engine_types.find(events_ptr->engineType);
      if (engine_type_it != engine_types.end()) {
        auto& engine_type = engine_type_it->second;
        engine_type.engines.push_back(
            {.index = events_ptr->engineIndex,
             .name = events_ptr->arguments.name});
        result.setLine(engine_type, events_ptr->engineIndex);
      }
    }
    return result;
  };
};

class HpuTraceParser {
 public:
  HpuTraceParser(
      std::deque<GenericTraceActivity>& activities,
      uint64_t hpu_start_time,
      uint64_t wall_start_time)
      : activities_{activities},
        hpu_start_time_{hpu_start_time},
        wall_start_time_{wall_start_time} {}

  virtual ~HpuTraceParser() = default;

  void Export(
      synTraceEvent* events_ptr,
      size_t num_events,
      uint64_t wall_stop_time) {
    engine_type_database_ =
        EngineDatabase::buildDatabase(events_ptr, num_events);
    initLanes();
    convertEventsToActivities(events_ptr, num_events, wall_stop_time);
  }

 private:
  bool SkipEvent(const synTraceEvent* events_ptr) {
    if (events_ptr->type == EventType::metadata)
      return true;

    auto& engine_types = engine_type_database_.engine_types_;
    if (engine_types.find(events_ptr->engineType) == engine_types.end())
      return true;

    absl::string_view name = events_ptr->name;
    if (name.find("write to mem") != absl::string_view::npos) {
      return true;
    }
    return false;
  }

  void initLanes() {
    addLane(plane_name_, device_lane_, 0, true, 0);
    auto& engine_types = engine_type_database_.engine_types_;
    for (auto& e : engine_types) {
      auto& engine_type = e.second;
      auto& engine_type_index = e.first;

      if (EngineType::isHost(engine_type.name)) {
        for (auto& engine : engine_type.engines) {
          addLane(
              std::string("Synapse/") + engine.name,
              engine_type_index,
              engine_type_database_.getLine(engine.index),
              false);
        }
      } else {
        for (auto& engine : engine_type.engines) {
          addLane(
              engine.name,
              device_lane_,
              engine_type_database_.getLine(engine.index),
              false);
        }
      }
    }
  }

  void convertEventsToActivities(
      synTraceEvent* events_ptr,
      size_t num_events,
      uint64_t wall_stop_time) {
    std::unordered_map<
        uint32_t,
        std::unordered_map<uint32_t, std::list<const synTraceEvent*>>>
        activeEvents;
    for (size_t i{}; i < num_events; i++, events_ptr++) {
      if (SkipEvent(events_ptr))
        continue;
      if (events_ptr->type == EventType::begin) {
        activeEvents[events_ptr->engineIndex][events_ptr->contextId].push_back(
            events_ptr);
      } else if (events_ptr->type == EventType::end) {
        auto& eventList =
            activeEvents[events_ptr->engineIndex][events_ptr->contextId];
        if (eventList.empty()) {
          // ShowWarning("END event without BEGIN", events_ptr);
        } else {
          auto start = normalizeTimeStamp(eventList.front()->timestamp);
          auto end = normalizeTimeStamp(events_ptr->timestamp);
          createTraceActivity(events_ptr, start, end);
        }
      } else if (events_ptr->type == EventType::complete) {
        auto start = normalizeTimeStamp(events_ptr->timestamp);
        if (start < wall_start_time_ || start > wall_stop_time) {
          // list might contain events before tracing start, ignore
          continue;
        } else {
          auto end =
              normalizeTimeStamp(events_ptr->timestamp + events_ptr->duration);
          createTraceActivity(events_ptr, start, end);
        }
      }
    }
  }

  void createTraceActivity(
      synTraceEvent* events_ptr,
      uint64_t start,
      uint64_t end) {
    auto isTPC = engine_type_database_.isEngineTypeTPC(events_ptr->engineType);
    GenericTraceActivity ev{
        defaultTraceSpan(),
        isTPC ? ActivityType::CONCURRENT_KERNEL : ActivityType::HPU_OP,
        StringOrFallback(events_ptr->arguments.operation, events_ptr->name)};
    ev.startTime = start;
    ev.endTime = end;
    ev.device = getDevice(events_ptr);
    ev.resource = engine_type_database_.getLine(events_ptr->engineIndex);
    if (isTPC) {
      ev.addMetadata("device", ev.device);
    }
    activities_.push_back(ev);
  }

  uint64_t normalizeTimeStamp(long double t) {
    auto result = static_cast<int64_t>(t) - hpu_start_time_ + wall_start_time_;
    return result > 0 ? result : 0;
  }

  void addLane(
      std::string name,
      int64_t pid,
      int64_t tid,
      bool is_process,
      int64_t sort_index = -1) {
    std::string label = is_process ? "process" : "thread";
    GenericTraceActivity name_meta{
        defaultTraceSpan(), ActivityType::HPU_META_OP, ""};
    name_meta.startTime = 0;
    name_meta.endTime = 0;
    name_meta.activityName = label + "_name";
    name_meta.device = pid;
    name_meta.resource = tid;
    name_meta.addMetadata("name", std::string("\"") + name + "\"");
    activities_.push_back(name_meta);
    if (sort_index != -1) {
      GenericTraceActivity sort_meta{
          defaultTraceSpan(), ActivityType::HPU_META_OP, ""};
      sort_meta.startTime = 0;
      sort_meta.endTime = 0;
      sort_meta.device = pid;
      sort_meta.resource = tid;
      sort_meta.activityName = "process_sort_index";
      sort_meta.addMetadata("sort_index", std::to_string(sort_index));
      activities_.push_back(sort_meta);
    }
  }

  const TraceSpan& defaultTraceSpan() {
    static TraceSpan span(0, 0, "PyTorch Profiler", "");
    return span;
  }

  int64_t getDevice(const synTraceEvent* events_ptr) {
    return engine_type_database_.isEngineTypeHost(events_ptr->engineType)
        ? events_ptr->engineType
        : device_lane_;
  }

  const std::string plane_name_ = "/device:HPU:0";
  std::deque<GenericTraceActivity>& activities_;
  uint64_t hpu_start_time_;
  uint64_t wall_start_time_;
  pid_t device_lane_{1};
  EngineDatabase engine_type_database_;
};

class ProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  explicit ProfilerSession(int64_t, int64_t) {
    status_ = TraceStatus::READY;
  }

  void start() override {
    doStart();
    status_ = TraceStatus::RECORDING;
  }

  void stop() override {
    doStop();
    status_ = TraceStatus::PROCESSING;
    convertLogs();
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
  void doStart() {
    // Necessary to initialize the device to use synapse api calls
    HABANAGuardImpl h;
    h.getDevice();
    auto hpu_start_time = nowNanos(CLOCK_MONOTONIC_RAW) / 1000;
    auto wall_start_time = nowNanos(CLOCK_REALTIME) / 1000;
    parser_ = std::make_unique<HpuTraceParser>(
        activities_, hpu_start_time, wall_start_time);

    synStatus status = synProfilerStart(synTraceAll, 0);
    if (status != synSuccess) {
      std::cerr << "synProfilerStart failed" << std::endl;
    }
  }

  void doStop() {
    synStatus status = synProfilerStop(synTraceAll, 0);
    if (status != synSuccess) {
      std::cerr << "synProfilerStop failed" << std::endl;
    }
  }

  void convertLogs() {
    activities_.clear();
    auto wall_stop_time = nowNanos(CLOCK_REALTIME) / 1000;

    size_t size{}, count{};
    getLogsSize(size, count);
    if (count == 0) {
      std::cerr << "No profiler entries" << std::endl;
      return;
    }
    std::vector<char> buf(size, 0);
    if (!getEntries(size, count, buf.data())) {
      return;
    }
    auto events = reinterpret_cast<synTraceEvent*>(buf.data());
    parser_->Export(events, count - 1, wall_stop_time);
  }

  void getLogsSize(size_t& size, size_t& count) {
    auto status = synProfilerGetTrace(
        synTraceAll, 0, synTraceFormatTEF, nullptr, &size, &count);
    if (status != synSuccess) {
      std::cerr << "synProfilerGetTrace failed" << std::endl;
    }
  }

  bool getEntries(size_t& size, size_t& count, void* out) {
    auto status = synProfilerGetTrace(
        synTraceAll, 0, synTraceFormatTEF, out, &size, &count);
    if (status != synSuccess) {
      std::cerr << "synProfilerGetTrace failed" << std::endl;
      return false;
    }
    return true;
  }

  std::deque<GenericTraceActivity> activities_;
  std::unique_ptr<HpuTraceParser> parser_;
};

class ActivityProfiler : public libkineto::IActivityProfiler {
 public:
  ActivityProfiler() {}
  virtual ~ActivityProfiler() override {}

  virtual const std::string& name() const override {
    return device_name;
  }

  virtual const std::set<ActivityType>& availableActivities() const override {
    return supported_activities;
  }

  virtual std::unique_ptr<IActivityProfilerSession> configure(
      const std::set<ActivityType>& activity_types,
      const KINETO_NAMESPACE::Config& config) override {
    auto start_time_ms =
        duration_cast<milliseconds>(system_clock::now().time_since_epoch())
            .count();
    return configure(start_time_ms, 0, activity_types, config);
  }

  virtual std::unique_ptr<IActivityProfilerSession> configure(
      int64_t start_time_ms,
      int64_t duration_ms,
      const std::set<ActivityType>& activity_types,
      const KINETO_NAMESPACE::Config&) override {
    auto env = std::getenv("HABANA_PROFILE");
    bool hpu_profiling_available =
        (env != nullptr) && (absl::string_view{env} != "0");
    bool hpu_profiling_requested =
        activity_types.find(ActivityType::HPU_OP) != activity_types.end() ||
        activity_types.find(ActivityType::HPU_META_OP) != activity_types.end();

    if (hpu_profiling_requested) {
      if (hpu_profiling_available) {
        auto session =
            std::make_unique<ProfilerSession>(start_time_ms, duration_ms);
        return session;
      } else {
        std::cerr
            << "Tensorboard callback for HPU hardware profiling disabled. To enable set \"HABANA_PROFILE\""
            << std::endl;
      }
    }
    return nullptr;
  }

 private:
  const std::set<ActivityType> supported_activities{
      ActivityType::HPU_OP,
      ActivityType::CONCURRENT_KERNEL,
      ActivityType::HPU_META_OP};
  std::string device_name{"HPU"};
};

std::unique_ptr<IActivityProfiler> register_activity_profiler() {
  return std::make_unique<ActivityProfiler>();
}

auto register_activity_profiler_factory = [] {
  libkineto::api().registerProfilerFactory(register_activity_profiler);
  return 0;
}();
}; // namespace habana
#undef FMT_HEADER_ONLY