#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include "nlohmann/json.hpp"
#include "synapse_profiler.h"

namespace habana {

class Parser {
 public:
  Parser() = default;

  void add_kernel_event(
      const std::string_view& name,
      int64_t pid,
      int64_t tid,
      int64_t ts,
      int64_t dur) {
    auto event = construct_event(name, "Kernel", pid, tid, ts, dur);
    event["args"]["device"] = pid;
    addToEvents(event);
  }

  void add_runtime_event(
      const std::string_view& name,
      int64_t pid,
      int64_t tid,
      int64_t ts,
      int64_t dur) {
    auto event = construct_event(name, "Runtime", pid, tid, ts, dur);
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

  void merge(const std::string_view& path) {
    nlohmannV340::json json_file;
    {
      std::ifstream i(static_cast<std::string>(path));
      i >> json_file;
    }
    auto& traceEvents = json_file["traceEvents"];
    traceEvents.insert(traceEvents.end(), json_.begin(), json_.end());
    {
      std::ofstream o(static_cast<std::string>(path));
      o << json_file;
    }
  }

 private:
  void addToEvents(const nlohmannV340::json& obj) {
    json_.push_back(obj);
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
  nlohmannV340::json json_;
};

class JsonActivityProfiler : public SynapseProfiler {
 public:
  JsonActivityProfiler() = default;

  void addActivity(
      const std::string& name,
      bool isKernel,
      int64_t device,
      int64_t resource,
      uint64_t start,
      uint64_t end) {
    if (start > 0 && end > 0) {
      if (isKernel) {
        parser_.add_kernel_event(name, device, resource, start, end - start);
      } else {
        parser_.add_runtime_event(name, device, resource, start, end - start);
      }
    }
  }

  void addDevice(const std::string& name, int64_t device) {
    parser_.add_device_info(name, name, device, 0);
  }

  void addResource(
      const std::string& name,
      int64_t device,
      int64_t resource,
      int64_t sort_index = -1) {
    parser_.add_resource_info(name, device, resource, sort_index, 0);
  }

  static JsonActivityProfiler* instance() {
    try {
      static JsonActivityProfiler this_;
      return &this_;
    } catch (std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
    }
    return nullptr;
  }

  static void exportProfilerLogs(const std::string_view& path) {
    auto profiler(instance());
    if (profiler)
      profiler->parser_.merge(path);
  }

  static void startProfilerSession() {
    auto profiler(instance());
    if (profiler)
      profiler->start();
  }

  static void stopProfilerSession() {
    auto profiler(instance());
    if (profiler)
      profiler->stop();
  }

  static uint64_t addCustomTagBegin(const std::string& tag) {
    auto profiler(instance());
    if (profiler)
      return profiler->startCustomMeasurement(tag);
    else
      return 0;
  }
  static void addCustomTagEnd(uint64_t id) {
    auto profiler(instance());
    if (profiler)
      return profiler->stopCustomMeasurement(id);
  }

 private:
  Parser parser_;
};

void export_profiler_logs(const std::string_view& path) {
  JsonActivityProfiler::exportProfilerLogs(path);
}
void start_profiler_session() {
  JsonActivityProfiler::startProfilerSession();
}
void stop_profiler_session() {
  JsonActivityProfiler::stopProfilerSession();
}
uint64_t add_custom_tag_begin(const std::string& tag) {
  return JsonActivityProfiler::addCustomTagBegin(tag);
}
void add_custom_tag_end(uint64_t id) {
  JsonActivityProfiler::addCustomTagEnd(id);
}
}; // namespace habana