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

#include <iostream>
#include <string>
#include <string_view>

#include "backend/profiling/json_file_parser.h"
#include "backend/profiling/profiling.h"

namespace habana {
namespace profile {

class JsonActivityProfiler : public Profiler {
 public:
  JsonActivityProfiler() : Profiler{parser_} {}
  virtual ~JsonActivityProfiler() {}

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

 private:
  JsonFileParser parser_;
};

void export_profiler_logs(std::string_view path) {
  JsonActivityProfiler::exportProfilerLogs(path);
}
void setup_profiler_sources(
    bool synapse_logger,
    bool bridge,
    bool memory,
    const std::vector<std::string>& mandatory_events) {
  JsonActivityProfiler::instance()->init_sources(
      synapse_logger, bridge, memory, mandatory_events);
}
void start_profiler_session() {
  JsonActivityProfiler::instance()->start();
}
void stop_profiler_session() {
  JsonActivityProfiler::instance()->stop();
}
}; // namespace profile
}; // namespace habana