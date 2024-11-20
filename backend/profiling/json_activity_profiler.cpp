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