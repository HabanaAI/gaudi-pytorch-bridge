#pragma once
#include <fstream>
#include <string>
#include <string_view>
#include "nlohmann/json.hpp"
#include "synapse_profiler.h"

namespace habana {

class Parser : public HPUDetailsConsumer {
 public:
  Parser() = default;

  void add_event(
      const std::string_view& name,
      ActivityType category,
      int64_t pid,
      int64_t tid,
      int64_t ts,
      int64_t dur) {
    auto event = construct_event(
        name, mapActivityTypeToString(category), pid, tid, ts, dur);
    if (category == ActivityType::KERNEL) {
      event["args"]["device"] = pid;
    }
    addToEvents(event);
  }

  void add_device_info(
      const std::string_view& name,
      const std::string_view& label,
      int64_t id,
      uint64_t time) {
    nlohmannV340::json process_name;
    process_name["name"] = "process_name";
    process_name["ph"] = "M";
    process_name["ts"] = time;
    process_name["pid"] = id;
    process_name["tid"] = 0;
    process_name["args"]["name"] = name;

    nlohmannV340::json process_labels;
    process_labels["name"] = "process_labels";
    process_labels["ph"] = "M";
    process_labels["ts"] = time;
    process_labels["pid"] = id;
    process_labels["tid"] = 0;
    process_labels["args"]["labels"] = label;

    nlohmannV340::json process_sort_index;
    process_sort_index["name"] = "process_sort_index";
    process_sort_index["ph"] = "M";
    process_sort_index["ts"] = time;
    process_sort_index["pid"] = id;
    process_sort_index["tid"] = 0;
    process_sort_index["args"]["sort_index"] = id < 8 ? id + 0x1000000ll : id;

    addToEvents(process_name);
    addToEvents(process_labels);
    addToEvents(process_sort_index);
  }

  void add_resource_info(
      const std::string_view& name,
      int64_t deviceId,
      int64_t id,
      int64_t sortIndex,
      uint64_t time) {
    nlohmannV340::json thread_name;
    thread_name["name"] = "thread_name";
    thread_name["ph"] = "M";
    thread_name["ts"] = time;
    thread_name["pid"] = deviceId;
    thread_name["tid"] = id;
    thread_name["args"]["name"] = name;

    nlohmannV340::json thread_sort_index;
    thread_sort_index["name"] = "thread_sort_index";
    thread_sort_index["ph"] = "M";
    thread_sort_index["ts"] = time;
    thread_sort_index["pid"] = deviceId;
    thread_sort_index["tid"] = id;
    thread_sort_index["args"]["sort_index"] = sortIndex;

    addToEvents(thread_name);
    addToEvents(thread_sort_index);
  }

  virtual void add_device_details(
      const std::unordered_map<std::string, std::string>& device_details) {
    nlohmannV340::json device_property;
    for (auto& key_value : device_details) {
      device_property[key_value.first] = key_value.second;
    }
    deviceProperties_.push_back(device_property);
  }

  nlohmannV340::json& get_create_array(
      nlohmannV340::json& json_file,
      const std::string_view& name) {
    if (json_file.find(name) == json_file.end())
      json_file[(std::string)name] = nlohmannV340::json::array();
    return json_file[(std::string)name];
  }

  void merge(const std::string_view& path) {
    nlohmannV340::json json_file;
    {
      std::ifstream i(static_cast<std::string>(path));
      i >> json_file;
    }
    if (!traceEvents_.empty()) {
      auto& traceEvents = get_create_array(json_file, "traceEvents");
      traceEvents.insert(
          traceEvents.end(), traceEvents_.begin(), traceEvents_.end());
    }

    if (!deviceProperties_.empty()) {
      auto& deviceProperties = get_create_array(json_file, "deviceProperties");
      deviceProperties.insert(
          deviceProperties.end(),
          deviceProperties_.begin(),
          deviceProperties_.end());
    }

    {
      std::ofstream o(static_cast<std::string>(path));
      o << json_file;
    }
  }

 private:
  void addToEvents(const nlohmannV340::json& obj) {
    traceEvents_.push_back(obj);
  }
  nlohmannV340::json construct_event(
      const std::string_view& name,
      const std::string_view& cat,
      int64_t pid,
      int64_t tid,
      int64_t ts,
      int64_t dur) {
    nlohmannV340::json runtime;
    runtime["ph"] = "X";
    runtime["cat"] = cat;
    runtime["name"] = name;
    runtime["pid"] = pid;
    runtime["tid"] = tid;
    runtime["ts"] = ts;
    runtime["dur"] = dur;
    return runtime;
  }
  std::string mapActivityTypeToString(ActivityType type) {
    switch (type) {
      case ActivityType::KERNEL:
        return "Kernel";
      case ActivityType::RUNTIME:
        return "Runtime";
      case ActivityType::MEMCPY:
        return "Memcpy";
      case ActivityType::MEMSET:
        return "Memset";
    }
    return "Runtime";
  }
  nlohmannV340::json traceEvents_;
  nlohmannV340::json deviceProperties_;
};
}; // namespace habana