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
#include <absl/strings/internal/ostringstream.h>
#include <absl/strings/string_view.h>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string_view>

struct synTensorDescriptor;
namespace synapse_logger {

/// Specifies types of data dumped to log file
enum class data_dump_category : unsigned {
  SYNAPSE_API_CALL =
      (0x1) << 0, ///< Synapse API Calls (synLaunch, synTensorCreate etc.)
  CUSTOM_RUNTIME_TRACE_PROVIDER = (0x1)
      << 1, ///< Calls introduced to code by runtime tracing structures
  VAR_TENSOR_DATA = (0x1) << 16, ///< Host-memory data (non-const tensor data)
  CONST_TENSOR_DATA = (0x1) << 17, ///< Host-memory data (const tensor data)
};

bool logger_is_enabled(data_dump_category cat);
bool log_observer_is_enabled(std::string_view name);
void log(const absl::string_view payload);
void on_log(std::string_view name, std::string_view args, bool begin);

class ostr_int_t {
 public:
  ostr_int_t() : buffer_(), ostr_(&buffer_) {
    buffer_.reserve(4096);
  }
  ostr_int_t(ostr_int_t&) = delete;
  ostr_int_t(ostr_int_t&&) = delete;
  ostr_int_t& operator=(ostr_int_t&) = delete;
  ostr_int_t& operator=(ostr_int_t&&) = delete;
  void clear() {
    ostr_.clear();
    buffer_ = "";
  }
  std::string* str() {
    return ostr_.str();
  }

  std::string buffer_;
  absl::strings_internal::OStringStream ostr_;
};
class ostr_t;
ostr_t get_ostr();

class ostr_t {
 public:
  ostr_t(ostr_t&& other) noexcept {
    std::swap(ostr_int_, other.ostr_int_);
  }

  ostr_t() = delete;
  ostr_t(const ostr_t&) = delete;
  ostr_t& operator=(const ostr_t& other) = delete;
  ostr_t& operator=(ostr_t&& other) noexcept {
    std::swap(ostr_int_, other.ostr_int_);
    return *this;
  }
  operator std::ostream &() {
    return ostr_int_->ostr_;
  }
  std::string& str() {
    return *ostr_int_->str();
  }
  ~ostr_t() {
    if (ostr_int_) {
      ostr_int_->clear();
    }
  }

 private:
  ostr_t(ostr_int_t& ostr_int) : ostr_int_(&ostr_int) {}
  ostr_int_t* ostr_int_{};
  friend ostr_t get_ostr();
  template <typename T>
  friend std::ostream& operator<<(ostr_t& out, T&& v);
};

template <typename T>
inline std::ostream& operator<<(ostr_t& out, T&& v) {
  out.ostr_int_->ostr_ << v;
  return out.ostr_int_->ostr_;
}

inline __attribute__((noinline)) ostr_t get_ostr() {
  thread_local ostr_int_t ostr;
  return ostr_t(ostr);
}

inline const absl::string_view type_name_from_pretty_function(
    const char* pretty_function) {
  const absl::string_view func(pretty_function);
  size_t pos = func.find("Type = ");
  return func.substr(pos + 7, func.length() - pos - 8);
};

template <typename Type>
inline void dump_object(const Type* obj, unsigned count = 1) {
  if (!logger_is_enabled(data_dump_category::SYNAPSE_API_CALL) ||
      obj == nullptr) {
    return;
  }
  synapse_logger::ostr_t out{synapse_logger::get_ostr()};
  static auto type_name{type_name_from_pretty_function(__PRETTY_FUNCTION__)};

  out << R"("name":"object", "args":{"at":")" << (void*)obj << R"(", "type":")"
      << type_name << R"(", "size":)" << (sizeof(Type) * count)
      << R"(, "value":")" << std::hex << "{0x";
  uint8_t* buffer = (uint8_t*)obj;

  unsigned p;
  for (p = 0; p < sizeof(Type) * count; ++p) {
    out << unsigned(buffer[p]) << ",0x";
  }
  out << unsigned(buffer[p]) << "}\"}" << std::dec;

  log(out.str());
}

} // namespace synapse_logger
