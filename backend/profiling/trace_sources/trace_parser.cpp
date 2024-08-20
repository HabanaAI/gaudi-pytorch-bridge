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

#include "backend/profiling/trace_sources/trace_parser.h"
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <list>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace habana {
namespace profile {

using namespace std::chrono;

enum EventType { begin = 'B', end = 'E', metadata = 'M', complete = 'X' };

const char* StringOrFallback(const char* main, const char* fallback) {
  return (main == nullptr or std::strlen(main) == 0) ? fallback : main;
}

bool Contains(const char* haystack, const char* needle) {
  return (
      haystack != nullptr && needle != nullptr &&
      std::strstr(haystack, needle) != nullptr);
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
        "*NIC External ",
        "*NIC Internal ",
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
  EngineDatabase(unsigned offset) : offset_{offset} {}

  std::unordered_map<uint32_t, EngineType> engine_types_;
  std::unordered_map<uint32_t, uint32_t> line_info_;
  unsigned offset_{};

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
    line_info_[index] = idx * seperator + index + offset_;
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
      synTraceEvent* events_ptr,
      size_t num_events,
      unsigned offset) {
    std::unique_ptr<EngineDatabase> result =
        std::make_unique<EngineDatabase>(offset);
    auto& engine_types = result->engine_types_;

    // Create host engine type first
    synTraceEvent* host_meta_event_ptr = events_ptr;
    for (uint64_t i = 0; i < num_events; i++, host_meta_event_ptr++) {
      if (host_meta_event_ptr->type != EventType::metadata)
        break;
      if (host_meta_event_ptr->engineIndex == 0 &&
          EngineType::isHost(host_meta_event_ptr->arguments.name)) {
        auto& engine_type = engine_types[host_meta_event_ptr->engineType];
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
    long double hpu_start_time,
    long double wall_start_time,
    unsigned offset)
    : hpu_start_time_{hpu_start_time},
      wall_start_time_{wall_start_time},
      offset_{offset} {}

HpuTraceParser::~HpuTraceParser() {}

void HpuTraceParser::Export(
    synTraceEvent* events_ptr,
    size_t num_events,
    long double wall_stop_time,
    TraceSink& trace_sink) {
  engine_type_database_ =
      EngineDatabase::buildDatabase(events_ptr, num_events, offset_);
  initLanes(trace_sink);
  convertEventsToActivities(events_ptr, num_events, wall_stop_time, trace_sink);
}

bool HpuTraceParser::skipEvent(const synTraceEvent* events_ptr) {
  if (events_ptr->type == EventType::metadata)
    return true;

  auto& engine_types = engine_type_database_->engine_types_;
  if (engine_types.find(events_ptr->engineType) == engine_types.end())
    return true;

  std::string_view name = events_ptr->name;
  if (name.find("write to mem") != std::string_view::npos) {
    return true;
  }
  return false;
}

void HpuTraceParser::initLanes(TraceSink& trace_sink) {
  trace_sink.addDevice(plane_name_, device_lane_);
  auto& engine_types = engine_type_database_->engine_types_;
  for (auto& e : engine_types) {
    auto& engine_type = e.second;
    auto& engine_type_index = e.first;

    if (EngineType::isHost(engine_type.name)) {
      for (auto& engine : engine_type.engines) {
        trace_sink.addResource(
            std::string("Synapse/") + engine.name,
            engine_type_index,
            engine_type_database_->getLine(engine.index));
      }
    } else {
      for (auto& engine : engine_type.engines) {
        trace_sink.addResource(
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
  start *= 1000;
  end *= 1000;
  return start > hpu_start_time_ && end < wall_stop_time;
}

void HpuTraceParser::processActivity(
    long double event_start_time,
    long double event_end_time,
    long double wall_stop_time,
    synTraceEvent* events_ptr,
    synTraceEvent* enqueue_events_ptr,
    TraceSink& trace_sink) {
  if (isEventInTime(event_start_time, event_end_time, wall_stop_time)) {
    auto start = timeStampHpuToTB(event_start_time);
    auto end = timeStampHpuToTB(event_end_time);
    std::string name =
        StringOrFallback(events_ptr->arguments.operation, events_ptr->name);

    trace_sink.addCompleteActivity(
        {name,
         getExtraArgs(events_ptr),
         getActivityType(events_ptr),
         getDevice(events_ptr),
         engine_type_database_->getLine(events_ptr->engineIndex)},
        std::make_optional<RecipeInfo>(
            {events_ptr->arguments.recipeId,
             events_ptr->arguments.recipeName,
             events_ptr->arguments.streamHandle,
             events_ptr->arguments.eventHandle}),
        start,
        end);

    if (enqueue_events_ptr != nullptr && enqueue_events_ptr != events_ptr) {
      trace_sink.addFlowEvent(
          name,
          "enqueue", // category
          {getDevice(enqueue_events_ptr),
           engine_type_database_->getLine(enqueue_events_ptr->engineIndex),
           timeStampHpuToTB(enqueue_events_ptr->timestamp)},
          {getDevice(events_ptr),
           engine_type_database_->getLine(events_ptr->engineIndex),
           start});
    }
  }
}

void HpuTraceParser::convertEventsToActivities(
    synTraceEvent* events_ptr,
    size_t num_events,
    long double wall_stop_time,
    TraceSink& trace_sink) {
  struct ActiveEvent {
    synTraceEvent* begin_;
    synTraceEvent* enqueue_;
  };
  using ActiveEventsMap = std::unordered_map<
      uint32_t,
      std::unordered_map<uint32_t, std::list<ActiveEvent>>>;
  using ActiveEnqueueEventsMap = std::unordered_map<uint32_t, synTraceEvent*>;
  ActiveEventsMap activeEvents;
  ActiveEnqueueEventsMap activeEnqueueEvents;

  for (size_t i{}; i < num_events; i++, events_ptr++) {
    if (skipEvent(events_ptr)) {
      continue;
    }

    if (events_ptr->arguments.recipeId != 0 &&
        Contains(events_ptr->name, "enqueueWithExternalEvents")) {
      activeEnqueueEvents[events_ptr->arguments.recipeId] = events_ptr;
    }

    switch (events_ptr->type) {
      case EventType::begin: {
        // store begin info, don't add anything
        auto& eventList =
            activeEvents[events_ptr->engineIndex][events_ptr->contextId];
        ActiveEvent activeEvent{
            events_ptr, activeEnqueueEvents[events_ptr->arguments.recipeId]};
        eventList.push_back(activeEvent);
      } break;
      case EventType::end: {
        // combine begin info with this event and activity
        auto& eventList =
            activeEvents[events_ptr->engineIndex][events_ptr->contextId];
        if (!eventList.empty()) {
          processActivity(
              eventList.front().begin_->timestamp,
              events_ptr->timestamp,
              wall_stop_time,
              events_ptr,
              eventList.front().enqueue_,
              trace_sink);
          eventList.pop_front();
        }
      } break;
      case EventType::complete: {
        processActivity(
            events_ptr->timestamp,
            events_ptr->timestamp + events_ptr->duration,
            wall_stop_time,
            events_ptr,
            activeEnqueueEvents[events_ptr->arguments.recipeId],
            trace_sink);
      } break;
    }
  }
}

int64_t HpuTraceParser::timeStampHpuToTB(long double t) {
  t *= 1000;
  if (t > hpu_start_time_) {
    return static_cast<int64_t>(roundl(t - hpu_start_time_ + wall_start_time_));
  } else {
    return 0;
  }
}

int64_t HpuTraceParser::getDevice(const synTraceEvent* events_ptr) {
  return engine_type_database_->isEngineTypeHost(events_ptr->engineType)
      ? events_ptr->engineType
      : device_lane_;
}

bool HpuTraceParser::isEventKernel(const synTraceEvent* events_ptr) {
  return engine_type_database_->isKernelWhitelist(
             events_ptr->arguments.operation) ||
      engine_type_database_->isEngineTypeMME(events_ptr->engineType) ||
      engine_type_database_->isEngineTypeTPC(events_ptr->engineType);
}

ActivityType HpuTraceParser::getActivityType(const synTraceEvent* events_ptr) {
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

std::unordered_map<std::string, std::string> HpuTraceParser::getExtraArgs(
    const synTraceEvent* events_ptr) {
  std::unordered_map<std::string, std::string> extraArgs;
  extraArgs.reserve(events_ptr->arguments.extraArgs.count + 1);

  extraArgs["dataType"] = events_ptr->arguments.dataType;

  for (size_t i{}; i < events_ptr->arguments.extraArgs.count; i++) {
    const auto& arg = events_ptr->arguments.extraArgs.args[i];
    std::string valueStr;

    switch (arg.type) {
      case synTraceEventArg::TYPE_CHAR_PTR:
        valueStr = arg.value.str;
        break;
      case synTraceEventArg::TYPE_UINT64:
        valueStr = std::to_string(arg.value.u64);
        break;
      case synTraceEventArg::TYPE_DOUBLE:
        valueStr = std::to_string(arg.value.d);
        break;
      default:
        break;
    }

    if (!valueStr.empty()) {
      extraArgs[arg.key] = valueStr;
    }
  }
  return extraArgs;
}
}; // namespace profile
}; // namespace habana