
#include <iostream>
#include <string>
#include <string_view>

#include "pytorch_helpers/habana_helpers/profiling/json_file_parser.h"
#include "pytorch_helpers/habana_helpers/profiling/profiling.h"

namespace habana {

class JsonActivityProfiler : public Profiler {
 public:
  JsonActivityProfiler() : Profiler{parser_} {}

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

 private:
  JsonFileParser parser_;
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
}; // namespace habana