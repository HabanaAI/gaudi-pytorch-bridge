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

#include <strings.h>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "backend/profiling/profiling.h"
#include "pytorch_helpers/synapse_logger/synapse_logger_observer.h"

namespace habana {
namespace profile {

class SynapseLoggerSource : public TraceSource,
                            public synapse_logger::SynapseLoggerObserver {
 public:
  SynapseLoggerSource(
      bool is_active,
      const std::vector<std::string>& mandatory_events);
  virtual ~SynapseLoggerSource() = default;
  void start(TraceSink&) override;
  void stop() override;
  void extract(TraceSink& trace_sink) override;
  TraceSourceVariant get_variant() override;
  void set_offset(unsigned offset) override;

  void on_log(
      std::string_view name,
      std::string_view args,
      pid_t pid,
      pid_t tid,
      int64_t dtime,
      bool begin) override;

  virtual bool enabled(std::string_view name = "") override;

 private:
  bool exists_on_mandatory_list(std::string_view name = "");

  struct Event {
    std::string name;
    std::string args;
    pid_t pid;
    pid_t tid;
    int64_t time;
    bool begin;

    Event(
        std::string&& name,
        std::string&& args,
        pid_t pid,
        pid_t tid,
        int64_t time,
        bool begin)
        : name(std::move(name)),
          args(std::move(args)),
          pid(pid),
          tid(tid),
          time(time),
          begin(begin) {}
  };
  std::deque<Event> events_;
  unsigned offset_{};
  std::mutex m{};
  bool is_started_{false};
  bool catch_all_events_{false};
  std::vector<std::string> mandatory_events_;
  struct {
    std::unordered_map<const char*, bool> go;
    std::mutex m{};
  } checked_;
};
} // namespace profile
} // namespace habana