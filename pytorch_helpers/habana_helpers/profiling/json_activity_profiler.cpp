#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include "json_parser.h"
#include "nlohmann/json.hpp"
#include "synapse_profiler.h"

namespace habana {

class JsonActivityProfiler : public SynapseProfiler {
 public:
  JsonActivityProfiler() : SynapseProfiler(parser_) {}

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