#pragma once
#include <fstream>
#include <string>
#include <string_view>
#include "nlohmann/json.hpp"
#include "pytorch_helpers/habana_helpers/profiling/profiling.h"

namespace habana {

class JsonFileParser : public TraceSink {
 public:
  JsonFileParser() = default;

  void addActivity(
      Activity activity,
      const std::optional<RecipeInfo>& recipeInfo,
      uint64_t time,
      bool begin) override {
    if (time > 0) {
      auto event = constructEvent(activity, recipeInfo, time);
      event["ph"] = begin ? "B" : "E";
      addToEvents(event);
    }
  }

  void addCompleteActivity(
      const Activity& activity,
      const std::optional<RecipeInfo>& recipeInfo,
      uint64_t start,
      uint64_t end) override {
    if (start > 0 && end > 0) {
      auto event = constructEvent(activity, recipeInfo, start);
      event["ph"] = "X";
      event["dur"] = (int64_t)(end - start);
      addToEvents(event);
    }
  }

  void addFlowEvent(
      const std::string_view& name,
      const std::string_view& cat,
      const Flow& start,
      const Flow& finish) {
    auto flow_start = construct_flow(
        name, cat, start.device, start.resource, start.time, true);
    auto flow_end = construct_flow(
        name, cat, finish.device, finish.resource, finish.time, false);
    addToEvents(flow_start);
    addToEvents(flow_end);
  }

  void addDevice(const std::string_view& name, int64_t id) override {
    nlohmannV340::json process_name;
    process_name["name"] = "process_name";
    process_name["ph"] = "M";
    process_name["ts"] = 0;
    process_name["pid"] = id;
    process_name["tid"] = 0;
    process_name["args"]["name"] = name;

    nlohmannV340::json process_labels;
    process_labels["name"] = "process_labels";
    process_labels["ph"] = "M";
    process_labels["ts"] = 0;
    process_labels["pid"] = id;
    process_labels["tid"] = 0;
    process_labels["args"]["labels"] = name;

    nlohmannV340::json process_sort_index;
    process_sort_index["name"] = "process_sort_index";
    process_sort_index["ph"] = "M";
    process_sort_index["ts"] = 0;
    process_sort_index["pid"] = id;
    process_sort_index["tid"] = 0;
    process_sort_index["args"]["sort_index"] = id < 8 ? id + 0x1000000ll : id;

    addToEvents(process_name);
    addToEvents(process_labels);
    addToEvents(process_sort_index);
  }

  void addResource(
      const std::string_view& name,
      int64_t deviceId,
      int64_t id,
      int64_t sortIndex) override {
    nlohmannV340::json thread_name;
    thread_name["name"] = "thread_name";
    thread_name["ph"] = "M";
    thread_name["ts"] = 0;
    thread_name["pid"] = deviceId;
    thread_name["tid"] = id;
    thread_name["args"]["name"] = name;

    nlohmannV340::json thread_sort_index;
    thread_sort_index["name"] = "thread_sort_index";
    thread_sort_index["ph"] = "M";
    thread_sort_index["ts"] = 0;
    thread_sort_index["pid"] = deviceId;
    thread_sort_index["tid"] = id;
    thread_sort_index["args"]["sort_index"] = sortIndex;

    addToEvents(thread_name);
    addToEvents(thread_sort_index);
  }

  virtual void addDeviceDetails(
      const std::unordered_map<std::string, std::string>& device_details)
      override {
    nlohmannV340::json device_property;
    for (auto& key_value : device_details) {
      device_property[key_value.first] = key_value.second;
    }
    deviceProperties_.push_back(device_property);
  }

  nlohmannV340::json& getCreateArray(
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
      auto& traceEvents = getCreateArray(json_file, "traceEvents");
      traceEvents.insert(
          traceEvents.end(), traceEvents_.begin(), traceEvents_.end());
    }

    if (!deviceProperties_.empty()) {
      auto& deviceProperties = getCreateArray(json_file, "deviceProperties");
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

  nlohmannV340::json constructEvent(
      const Activity& activity,
      const std::optional<RecipeInfo>& recipeInfo,
      int64_t ts) {
    nlohmannV340::json runtime;

    runtime["cat"] = mapActivityTypeToString(activity.type),
    runtime["name"] = activity.name;
    runtime["pid"] = activity.device;
    runtime["tid"] = activity.resource;
    runtime["ts"] = ts;
    if (activity.func != nullptr) {
      runtime["func"] = activity.func;
    }

    nlohmannV340::json args;

    if (recipeInfo) {
      args["recipeId"] = recipeInfo->recipeId;
      args["recipeName"] = recipeInfo->recipeName;
      args["streamHandle"] = recipeInfo->streamHandle;
      args["eventHandle"] = recipeInfo->eventHandle;
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
  nlohmannV340::json construct_flow(
      const std::string_view& name,
      const std::string_view& cat,
      int64_t pid,
      int64_t tid,
      int64_t ts,
      bool start) {
    nlohmannV340::json flow;
    flow["ph"] = start ? "s" : "f";
    flow["cat"] = cat;
    flow["name"] = name;
    flow["ts"] = ts;
    flow["pid"] = pid;
    flow["tid"] = tid;
    flow["bp"] = "e"; // if binding point is not set to enclosing slice ("e")
                      // flow will end in the first event after timestamp
    flow["id"] = flow_id_counter;
    if (!start)
      flow_id_counter++;
    return flow;
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
  uint64_t flow_id_counter = 0;
  nlohmannV340::json traceEvents_;
  nlohmannV340::json deviceProperties_;
};
}; // namespace habana