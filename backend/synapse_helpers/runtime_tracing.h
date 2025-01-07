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
#include <mutex>

#include "absl/strings/str_cat.h"
#include "synapse_logger/object_dump.h"
namespace synapse_helpers {
struct trace_provider_t {
  synapse_logger::data_dump_category id_;
};

inline bool trace_is_enabled_for_provider(trace_provider_t provider) {
  return synapse_logger::logger_is_enabled(provider.id_);
}

const trace_provider_t PROVIDER{
    synapse_logger::data_dump_category::CUSTOM_RUNTIME_TRACE_PROVIDER};

class trace_waitable_wait {
 public:
  trace_waitable_wait(const char* name, trace_provider_t provider = PROVIDER) {
    if (!trace_is_enabled_for_provider(provider)) {
      return;
    }
    synapse_logger::log(
        absl::StrCat(R"("name":")", name, R"(-wait", "cat":"sem", "ph":"B")"));
  }
};

inline void trace_waitable_acquire(
    const char* id,
    trace_provider_t provider = PROVIDER) {
  if (!trace_is_enabled_for_provider(provider)) {
    return;
  }
  synapse_logger::log(
      absl::StrCat(R"("name":")", id, R"(-wait", "cat":"sem", "ph":"E")"));
  synapse_logger::log(
      absl::StrCat(R"("name":")", id, R"(-acquire", "cat":"sem", "ph":"B")"));
}

inline void trace_waitable_release(
    const char* id,
    trace_provider_t provider = PROVIDER) {
  if (!trace_is_enabled_for_provider(provider)) {
    return;
  }
  synapse_logger::log(
      absl::StrCat(R"("name":")", id, R"(-acquire", "cat":"sem", "ph":"E")"));
}

inline void trace_start(const char* id, trace_provider_t provider = PROVIDER) {
  if (!trace_is_enabled_for_provider(provider)) {
    return;
  }
  {
    synapse_logger::log(
        absl::StrCat(R"("name":")", id, R"(", "cat":"foo", "ph":"B")"));
  }
}

inline void trace_event(const char* id, trace_provider_t provider = PROVIDER) {
  if (!trace_is_enabled_for_provider(provider)) {
    return;
  }
  synapse_logger::log(
      absl::StrCat(R"("name":")", id, R"(", "cat":"foo", "ph":"I")"));
}

inline void trace_end(const char* id, trace_provider_t provider = PROVIDER) {
  if (!trace_is_enabled_for_provider(provider)) {
    return;
  }
  synapse_logger::log(
      absl::StrCat(R"("name":")", id, R"(", "cat":"foo", "ph":"E")"));
}

class trace_scope_entry {
 public:
  trace_scope_entry(const char* name, trace_provider_t provider = PROVIDER) {
    trace_start(name, provider);
  }
};

class trace_scope {
 public:
  trace_scope(const char* name, trace_provider_t provider = PROVIDER)
      : name_(name), provider_(provider) {
    trace_start(name, provider);
  }
  ~trace_scope() {
    trace_end(name_, provider_);
  }

 private:
  const char* name_;
  trace_provider_t provider_;
};

template <typename T>
class trace_waitable {
 public:
  const char* name_;
  trace_waitable_wait entry_;
  T lock_;
  template <typename... Args>
  trace_waitable(const char* name, Args&&... args)
      : name_(name), entry_(name), lock_(std::forward<Args>(args)...) {
    trace_waitable_acquire(name);
  }
  ~trace_waitable() {
    trace_waitable_release(name_);
  }
};

template <typename T>
using tracing_lock_guard = trace_waitable<std::lock_guard<T>>;

} // namespace synapse_helpers
