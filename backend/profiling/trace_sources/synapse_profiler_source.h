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

#pragma once

#include <unordered_map>
#include "backend/profiling/profiling.h"
#include "backend/profiling/trace_sources/trace_parser.h"

namespace habana {
namespace profile {

class SynapseProfilerSource : public TraceSource {
 public:
  SynapseProfilerSource();
  ~SynapseProfilerSource() = default;

  void start(TraceSink& output) override;
  void stop() override;
  void extract(TraceSink& output) override;
  TraceSourceVariant get_variant() override;
  void set_offset(unsigned offset) override;

 private:
  void convertLogs(TraceSink& output);
  void getLogsSize(size_t& size, size_t& count);
  bool getEntries(size_t& size, size_t& count, void* out);
  void initHpuDetails(TraceSink& output);

  std::unique_ptr<HpuTraceParser> parser_;
  long double wall_stop_time_;
  unsigned offset_{};
};
} // namespace profile
} // namespace habana