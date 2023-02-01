#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <list>
#include <memory>
#include <unordered_set>
#include "absl/strings/string_view.h"
#define FMT_HEADER_ONLY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "spdlog/common.h"
#include "spdlog/fmt/bundled/format.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#include "trace_parser.h"

namespace habana {

using namespace std::chrono;

enum EventType { begin = 'B', end = 'E', metadata = 'M', complete = 'X' };

const char* StringOrFallback(const char* main, const char* fallback) {
  return (main == nullptr or std::strlen(main) == 0) ? fallback : main;
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
        "*NIC ",
        "**NIC ",
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
    for (size_t i = 0; i < tpc_engines.size(); i++) {
      if (name.find(tpc_engines[i]) == 0) {
        return true;
      }
    }
    return false;
  }
  static bool isMME(const std::string_view name) {
    static std::vector<std::string> mme_engines = {"**MME ", "*MME "};
    for (int i = 0; i < (int)mme_engines.size(); i++) {
      if (name.find(mme_engines[i]) == 0) {
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

  bool isEngineTypeMME(uint32_t engine_type) {
    auto engine_type_it = engine_types_.find(engine_type);
    return engine_type_it != engine_types_.end() &&
        EngineType::isMME(engine_type_it->second.name);
  }

  bool isKernelWhitelist(const char* operatorName) {
    if (operatorName == nullptr or std::strlen(operatorName) == 0) {
      return false;
    }
    std::string operatorString = operatorName;
    static std::unordered_set<std::string> whitelisted_events = {
        "DmaTranspose"};
    return whitelisted_events.find(operatorString) != whitelisted_events.end();
  }
  static std::unique_ptr<EngineDatabase> buildDatabase(
      synTraceEvent2* events_ptr,
      size_t num_events) {
    std::unique_ptr<EngineDatabase> result = std::make_unique<EngineDatabase>();
    auto& engine_types = result->engine_types_;

    // Create host engine type first
    synTraceEvent2* host_meta_event_ptr = events_ptr;
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
        result->setLine(engine_type, events_ptr->engineIndex);
      }
    }
    return result;
  };
};

HpuTraceParser::HpuTraceParser(
    TraceOutput& trace_output,
    long double hpu_start_time,
    long double wall_start_time)
    : trace_output_{trace_output},
      hpu_start_time_{hpu_start_time},
      wall_start_time_{wall_start_time} {}

HpuTraceParser::~HpuTraceParser() {}

void HpuTraceParser::Export(
    synTraceEvent2* events_ptr,
    size_t num_events,
    long double wall_stop_time) {
  engine_type_database_ = EngineDatabase::buildDatabase(events_ptr, num_events);
  initLanes();
  convertEventsToActivities(events_ptr, num_events, wall_stop_time);
}

bool HpuTraceParser::skipEvent(const synTraceEvent2* events_ptr) {
  if (events_ptr->type == EventType::metadata)
    return true;

  auto& engine_types = engine_type_database_->engine_types_;
  if (engine_types.find(events_ptr->engineType) == engine_types.end())
    return true;

  absl::string_view name = events_ptr->name;
  if (name.find("write to mem") != absl::string_view::npos) {
    return true;
  }
  return false;
}

void HpuTraceParser::initLanes() {
  trace_output_.addDevice(plane_name_, device_lane_);
  auto& engine_types = engine_type_database_->engine_types_;
  for (auto& e : engine_types) {
    auto& engine_type = e.second;
    auto& engine_type_index = e.first;

    if (EngineType::isHost(engine_type.name)) {
      for (auto& engine : engine_type.engines) {
        trace_output_.addResource(
            std::string("Synapse/") + engine.name,
            engine_type_index,
            engine_type_database_->getLine(engine.index));
      }
    } else {
      for (auto& engine : engine_type.engines) {
        trace_output_.addResource(
            engine.name,
            device_lane_,
            engine_type_database_->getLine(engine.index));
      }
    }
  }
}

bool HpuTraceParser::isEventInTime(
    long double start,
    long double end,
    long double wall_stop_time) {
  return start > hpu_start_time_ && end < wall_stop_time;
}

void HpuTraceParser::convertEventsToActivities(
    synTraceEvent2* events_ptr,
    size_t num_events,
    long double wall_stop_time) {
  std::unordered_map<
      uint32_t,
      std::unordered_map<uint32_t, std::list<const synTraceEvent2*>>>
      activeEvents;
  for (size_t i{}; i < num_events; i++, events_ptr++) {
    if (!skipEvent(events_ptr)) {
      if (events_ptr->type == EventType::begin ||
          events_ptr->type == EventType::end) {
        auto& eventList =
            activeEvents[events_ptr->engineIndex][events_ptr->contextId];
        if (events_ptr->type == EventType::begin) {
          eventList.push_back(events_ptr);
        } else {
          if (!eventList.empty()) {
            auto start = timeStampHpuToTB(eventList.front()->timestamp);
            auto end = timeStampHpuToTB(events_ptr->timestamp);
            if (isEventInTime(
                    eventList.front()->timestamp,
                    events_ptr->timestamp,
                    wall_stop_time)) {
              trace_output_.addActivity(
                  StringOrFallback(
                      events_ptr->arguments.operation, events_ptr->name),
                  getActivityType(events_ptr),
                  getDevice(events_ptr),
                  engine_type_database_->getLine(events_ptr->engineIndex),
                  start,
                  end);
            }
            eventList.pop_front();
          }
        }
        continue;
      } else if (events_ptr->type == EventType::complete) {
        auto start = timeStampHpuToTB(events_ptr->timestamp);
        auto end =
            timeStampHpuToTB(events_ptr->timestamp + events_ptr->duration);
        if (isEventInTime(
                events_ptr->timestamp,
                events_ptr->timestamp + events_ptr->duration,
                wall_stop_time)) {
          trace_output_.addActivity(
              StringOrFallback(
                  events_ptr->arguments.operation, events_ptr->name),
              getActivityType(events_ptr),
              getDevice(events_ptr),
              engine_type_database_->getLine(events_ptr->engineIndex),
              start,
              end);
        }
      }
    }
  }
}

int64_t HpuTraceParser::timeStampHpuToTB(long double t) {
  if (t > hpu_start_time_) {
    return static_cast<int64_t>(roundl(t - hpu_start_time_ + wall_start_time_));
  } else {
    return 0;
  }
}

int64_t HpuTraceParser::getDevice(const synTraceEvent2* events_ptr) {
  return engine_type_database_->isEngineTypeHost(events_ptr->engineType)
      ? events_ptr->engineType
      : device_lane_;
}

bool HpuTraceParser::isEventKernel(const synTraceEvent2* events_ptr) {
  return engine_type_database_->isKernelWhitelist(
             events_ptr->arguments.operation) ||
      engine_type_database_->isEngineTypeMME(events_ptr->engineType) ||
      engine_type_database_->isEngineTypeTPC(events_ptr->engineType);
}

ActivityType HpuTraceParser::getActivityType(const synTraceEvent2* events_ptr) {
  std::string name =
      StringOrFallback(events_ptr->arguments.operation, events_ptr->name);

  if (isEventKernel(events_ptr))
    return ActivityType::KERNEL;
  if (name.find("memcpy") == 0)
    return ActivityType::MEMCPY;
  if (name.find("memset") == 0)
    return ActivityType::MEMSET;
  return ActivityType::RUNTIME;
}

}; // namespace habana
#undef FMT_HEADER_ONLY