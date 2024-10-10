/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
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
