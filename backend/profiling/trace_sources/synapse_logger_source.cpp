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

#include "synapse_logger_source.h"
#include <unordered_set>
#include "pytorch_helpers/synapse_shim/synapse_api_shim.h"

using namespace synapse_logger;

namespace habana {
namespace profile {

namespace {
void convert(
    std::string_view input,
    std::unordered_map<std::string, std::string>& output) {
  std::string key, value;
  std::string* acc{&key};
  for (auto it{input.begin()};; it++) {
    if (it == input.end() || *it == ',') {
      acc = &key;
      if (!key.empty()) {
        output[key] = value;
      }
      if (it == input.end()) {
        break;
      }
      key.clear();
      value.clear();
    } else if (*it == ':') {
      acc = &value;
    } else if (*it != ' ' && *it != '\"') {
      acc->push_back(*it);
    }
  }
}
} // namespace

SynapseLoggerSource::SynapseLoggerSource() {
  EnableSynapseApiLogger(this);
}

void SynapseLoggerSource::start() {
  enabled_ = true;
}

void SynapseLoggerSource::stop() {
  enabled_ = false;
}

void SynapseLoggerSource::extract(TraceSink& trace_sink) {
  std::unordered_set<pid_t> pids;
  for (auto event : events_) {
    std::unordered_map<std::string, std::string> args;
    convert(event.args.c_str(), args);
    trace_sink.addActivity(
        {event.name,
         args,
         ActivityType::RUNTIME,
         event.pid + offset_,
         event.tid},
        {},
        event.time,
        event.begin);
    pids.insert(event.pid + offset_);
  }
  for (auto pid : pids) {
    trace_sink.addDevice("Synapse Logger", pid);
  }
}

TraceSourceVariant SynapseLoggerSource::get_variant() {
  return TraceSourceVariant::SYNAPSE_LOGGER;
}
void SynapseLoggerSource::set_offset(unsigned offset) {
  offset_ = offset;
}

void SynapseLoggerSource::on_log(
    std::string_view name,
    std::string_view args,
    pid_t pid,
    pid_t tid,
    int64_t time,
    bool begin) {
  events_.emplace_back(
      static_cast<std::string>(name),
      static_cast<std::string>(args),
      pid,
      tid,
      time,
      begin);
}

bool SynapseLoggerSource::enabled() {
  return enabled_;
}
} // namespace profile
} // namespace habana