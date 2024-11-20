/**
* Copyright (c) 2021-2024 Intel Corporation
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

#include "json_file_parser.h"
#include <iomanip>
#include <sstream>

namespace habana {
namespace profile {

ChromeTraceBaseTime& ChromeTraceBaseTime::singleton() {
  static ChromeTraceBaseTime instance;
  return instance;
}

// The 'ts' field written into the json file has 19 significant digits,
// while a double can only represent 15-16 digits. By using relative time,
// other applications can accurately read the 'ts' field as a double.
// Use the program loading time as the baseline time.
inline int64_t transToRelativeTime(int64_t time) {
  // Sometimes after converting to relative time, it can be a few nanoseconds
  // negative. Since Chrome trace and json processing will throw a parser error,
  // guard this.
  int64_t res = time - ChromeTraceBaseTime::singleton().get();
  if (res < 0) {
    return 0;
  }
  return res;
}

void JsonFileParser::addActivity(
    const Activity& activity,
    const std::optional<RecipeInfo>& recipeInfo,
    uint64_t time,
    bool begin) {
  if (time > 0) {
    auto event =
        constructEvent(activity, recipeInfo, transToRelativeTime(time));
    event["ph"] = begin ? "B" : "E";
    addToEvents(event);
  }
}

void JsonFileParser::addCompleteActivity(
    const Activity& activity,
    const std::optional<RecipeInfo>& recipeInfo,
    uint64_t start,
    uint64_t end) {
  if (start > 0 && end > 0) {
    auto event =
        constructEvent(activity, recipeInfo, transToRelativeTime(start));
    event["ph"] = "X";
    event["dur"] = convertToMs(end - start);
    addToEvents(event);
  }
}

void JsonFileParser::addFlowEvent(
    std::string_view name,
    std::string_view cat,
    const Flow& start,
    const Flow& finish) {
  if (GET_ENV_FLAG_NEW(PT_TB_ENABLE_FLOW_EVENTS)) {
    auto flow_start = constructFlow(
        name,
        cat,
        start.device,
        start.resource,
        transToRelativeTime(start.time),
        true);
    auto flow_end = constructFlow(
        name,
        cat,
        finish.device,
        finish.resource,
        transToRelativeTime(finish.time),
        false);
    addToEvents(flow_start);
    addToEvents(flow_end);
  }
}

void JsonFileParser::addMemoryEvent(
    int64_t device,
    int64_t resource,
    int64_t time,
    uint64_t addr,
    int64_t bytes,
    int64_t device_id,
    int64_t device_type,
    uint64_t total_allocated,
    uint64_t total_reserved) {
  auto memory_event = constructMemoryEvent(
      device,
      resource,
      transToRelativeTime(time),
      addr,
      bytes,
      device_id,
      device_type,
      total_allocated,
      total_reserved);
  addToEvents(memory_event);
}

void JsonFileParser::addDevice(std::string_view name, int64_t id) {
  nlohmannV340::json process_name;
  process_name["name"] = "process_name";
  process_name["ph"] = "M";
  process_name["ts"] = 0.0;
  process_name["pid"] = id;
  process_name["tid"] = 0;
  process_name["args"]["name"] = name;

  nlohmannV340::json process_labels;
  process_labels["name"] = "process_labels";
  process_labels["ph"] = "M";
  process_labels["ts"] = 0.0;
  process_labels["pid"] = id;
  process_labels["tid"] = 0;
  process_labels["args"]["labels"] = name;

  nlohmannV340::json process_sort_index;
  process_sort_index["name"] = "process_sort_index";
  process_sort_index["ph"] = "M";
  process_sort_index["ts"] = 0.0;
  process_sort_index["pid"] = id;
  process_sort_index["tid"] = 0;
  process_sort_index["args"]["sort_index"] = id < 8 ? id + 0x1000000ll : id;

  addToEvents(process_name);
  addToEvents(process_labels);
  addToEvents(process_sort_index);
}

void JsonFileParser::addResource(
    std::string_view name,
    int64_t deviceId,
    int64_t id,
    int64_t sortIndex) {
  nlohmannV340::json thread_name;
  thread_name["name"] = "thread_name";
  thread_name["ph"] = "M";
  thread_name["ts"] = 0.0;
  thread_name["pid"] = deviceId;
  thread_name["tid"] = id;
  thread_name["args"]["name"] = name;

  nlohmannV340::json thread_sort_index;
  thread_sort_index["name"] = "thread_sort_index";
  thread_sort_index["ph"] = "M";
  thread_sort_index["ts"] = 0.0;
  thread_sort_index["pid"] = deviceId;
  thread_sort_index["tid"] = id;
  thread_sort_index["args"]["sort_index"] = sortIndex;

  addToEvents(thread_name);
  addToEvents(thread_sort_index);
}

void JsonFileParser::addDeviceDetails(
    const std::unordered_map<std::string, std::string>& device_details) {
  for (auto& key_value : device_details) {
    deviceProperties_[key_value.first] = key_value.second;
  }
}

void JsonFileParser::addDeviceDetails(
    const std::unordered_map<std::string, int64_t>& device_details) {
  for (auto& key_value : device_details) {
    deviceProperties_[key_value.first] = key_value.second;
  }
}

nlohmannV340::json& JsonFileParser::getCreateArray(
    nlohmannV340::json& json_file,
    const std::string_view& name) {
  if (json_file.find(name) == json_file.end())
    json_file[(std::string)name] = nlohmannV340::json::array();
  return json_file[(std::string)name];
}

void JsonFileParser::merge(const std::string_view& path) {
  nlohmannV340::json json_file;
  {
    std::ifstream i(static_cast<std::string>(path));
    i >> json_file;
  }
  if (!traceEvents_.empty()) {
    auto& traceEvents = getCreateArray(json_file, "traceEvents");
    traceEvents.insert(
        traceEvents.end(), traceEvents_.begin(), traceEvents_.end());
  }

  if (!deviceProperties_.empty()) {
    auto& deviceProperties = getCreateArray(json_file, "deviceProperties");
    deviceProperties.push_back(deviceProperties_);
  }

  {
    std::ofstream o(static_cast<std::string>(path));
    o << json_file;
  }

  // clear the object after each data dump to:
  // - reduce footprint
  // - avoid data duplication in start/stop loop usage model
  clear();
}

void JsonFileParser::clear() {
  deviceProperties_.clear();
  traceEvents_.clear();
  profiler_event_index_ = 0;
  flow_id_counter_ = 0;
}

int64_t JsonFileParser::transToRelativeTime(int64_t time) {
  return habana::profile::transToRelativeTime(time);
}

void JsonFileParser::addToEvents(const nlohmannV340::json& obj) {
  traceEvents_.push_back(obj);
}

std::string JsonFileParser::toHex(uint64_t handle) {
  std::stringstream stream;
  stream << "0x" << std::hex << handle;
  return stream.str();
}

double JsonFileParser::convertToMs(uint64_t value) {
  return static_cast<double>(value) / 1000.0;
}

nlohmannV340::json JsonFileParser::constructEvent(
    const Activity& activity,
    const std::optional<RecipeInfo>& recipeInfo,
    int64_t ts) {
  nlohmannV340::json runtime;

  runtime["cat"] = mapActivityTypeToString(activity.type),
  runtime["name"] = activity.name;
  runtime["pid"] = activity.device;
  runtime["tid"] = activity.resource;
  runtime["ts"] = convertToMs(ts);

  nlohmannV340::json args;

  if (recipeInfo) {
    args["recipeId"] = recipeInfo->recipeId;
    args["recipeName"] = recipeInfo->recipeName;
    args["streamHandle"] = toHex(recipeInfo->streamHandle);
    args["eventHandle"] = toHex(recipeInfo->eventHandle);
  }

  if (activity.type == ActivityType::KERNEL) {
    args["device"] = activity.device;
  }

  if (!activity.args.empty()) {
    for (auto kv : activity.args) {
      args[kv.first] = kv.second;
    }
  }

  if (!args.empty()) {
    runtime["args"] = args;
  }

  return runtime;
}
nlohmannV340::json JsonFileParser::constructFlow(
    std::string_view name,
    std::string_view cat,
    int64_t pid,
    int64_t tid,
    int64_t ts,
    bool start) {
  nlohmannV340::json flow;
  flow["ph"] = start ? "s" : "f";
  flow["cat"] = cat;
  flow["name"] = name;
  flow["ts"] = convertToMs(ts);
  flow["pid"] = pid;
  flow["tid"] = tid;
  flow["bp"] = "e"; // if binding point is not set to enclosing slice ("e")
                    // flow will end in the first event after timestamp
  flow["id"] = flow_id_counter_;
  if (!start)
    flow_id_counter_++;
  return flow;
}
nlohmannV340::json JsonFileParser::constructMemoryEvent(
    int64_t pid,
    int64_t tid,
    int64_t ts,
    uint64_t addr,
    int64_t bytes,
    int64_t device_id,
    int64_t device_type,
    uint64_t total_allocated,
    uint64_t total_reserved) {
  nlohmannV340::json memory_event;
  memory_event["cat"] =
      mapActivityTypeToString(ActivityType::CPU_INSTANT_EVENT);
  memory_event["name"] = "[memory]";
  memory_event["ph"] = "i";
  memory_event["pid"] = pid;
  memory_event["s"] = "t";
  memory_event["tid"] = tid;
  memory_event["ts"] = convertToMs(ts);

  profiler_event_index_++;
  nlohmannV340::json args;
  args["Addr"] = addr;
  args["Bytes"] = bytes;
  args["Device Id"] = device_id;
  args["Device Type"] = device_type;
  args["Profiler Event Index"] = profiler_event_index_;
  args["Total Allocated"] = total_allocated;
  args["Total Reserved"] = total_reserved;

  memory_event["args"] = args;
  return memory_event;
}

std::string JsonFileParser::mapActivityTypeToString(ActivityType type) {
  switch (type) {
    case ActivityType::KERNEL:
      return "Kernel";
    case ActivityType::RUNTIME:
      return "Runtime";
    case ActivityType::MEMCPY:
      return "Memcpy";
    case ActivityType::MEMSET:
      return "Memset";
    case ActivityType::CPU_INSTANT_EVENT:
      return "cpu_instant_event";
  }
  return "Runtime";
}

} // namespace profile
} // namespace habana