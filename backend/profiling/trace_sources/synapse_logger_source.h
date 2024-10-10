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