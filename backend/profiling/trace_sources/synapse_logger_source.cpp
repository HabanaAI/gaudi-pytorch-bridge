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

#include "synapse_logger_source.h"
#include <regex>
#include <unordered_set>
#include "pytorch_helpers/synapse_shim/synapse_api_shim.h"

using namespace synapse_logger;

namespace habana {
namespace profile {

namespace {
std::unordered_map<std::string, std::string> convert(std::string_view input) {
  std::unordered_map<std::string, std::string> output;
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
  return output;
}
} // namespace

SynapseLoggerSource::SynapseLoggerSource(
    bool is_active,
    const std::vector<std::string>& mandatory_events)
    : catch_all_events_(is_active), mandatory_events_{mandatory_events} {
  EnableSynapseApiLogger(this);
}

void SynapseLoggerSource::start(TraceSink&) {
  is_started_ = true;
}

void SynapseLoggerSource::stop() {
  is_started_ = false;
}

void SynapseLoggerSource::extract(TraceSink& trace_sink) {
  if (events_.empty()) {
    return;
  }
  std::unordered_set<pid_t> pids;
  std::lock_guard<std::mutex> lg{m};
  for (const auto& event : events_) {
    trace_sink.addActivity(
        {event.name,
         convert(event.args),
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
  events_.clear();
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
  if (enabled(name)) {
    std::string event_name{name};
    std::string event_args{args};
    std::lock_guard<std::mutex> lg{m};
    events_.emplace_back(
        std::move(event_name), std::move(event_args), pid, tid, time, begin);
  }
}

bool SynapseLoggerSource::enabled(std::string_view name) {
  if (is_started_) {
    return catch_all_events_ || exists_on_mandatory_list(name);
  }
  return false;
}

bool SynapseLoggerSource::exists_on_mandatory_list(std::string_view name) {
  decltype(checked_.go)::iterator it_checked{};
  {
    std::lock_guard<std::mutex> lg{checked_.m};
    it_checked = checked_.go.find(name.data());
  }
  if (it_checked != checked_.go.end()) {
    return it_checked->second;
  }
  bool matched{false};
  for (const auto& mandatory_event : mandatory_events_) {
    std::regex mandatory_event_regex(
        mandatory_event, std::regex_constants::ECMAScript);
    if (std::regex_search(name.begin(), name.end(), mandatory_event_regex)) {
      matched = true;
      break;
    }
  }
  std::lock_guard<std::mutex> lg{checked_.m};
  checked_.go.emplace(name.data(), matched);
  return matched;
}
} // namespace profile
} // namespace habana