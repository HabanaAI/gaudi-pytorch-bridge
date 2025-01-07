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
#include <absl/types/span.h>
#include <iostream>
#include <sstream>
#include <tuple>

#include "object_dump.h"
#include "synapse_logger.h"

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const T* v) {
  return out << '"' << (const void*)(v) << '"';
}

inline std::ostream& operator<<(std::ostream& out, const char* v) {
  return std::operator<<(out, v);
}

template <typename T>
inline std::ostream& operator<<(
    std::ostream& stream,
    const absl::Span<T>& buffer) {
  if (buffer.size() && buffer.begin()) {
    for (unsigned long i = 0; i < buffer.size() - 1; ++i) {
      stream << buffer[i] << ", ";
    }
    stream << buffer[buffer.size() - 1];
  }
  return stream;
}

template <>
inline std::ostream& operator<<(
    std::ostream& stream,
    const absl::Span<int8_t>& buffer) {
  if (buffer.size() && buffer.begin()) {
    for (unsigned long i = 0; i < buffer.size() - 1; ++i) {
      stream << int(buffer[i]) << ", ";
    }
    stream << int(buffer[buffer.size() - 1]);
  }
  return stream;
}
template <typename... Args>
inline void comma_maybe(std::ostream& out, [[maybe_unused]] Args&&... args) {
  out << ", ";
}

inline void comma_maybe([[maybe_unused]] std::ostream& out) {}

enum arg_print_way {
  print_direct,
  print_quote,
  print_deref,
  print_quote_deref,
  print_array,
  print_hex,
  print_hex_array,
  print_hex_deref,
  print_quote_array
};

template <arg_print_way Way, typename V>
struct _argument_t {
  _argument_t(const char* name, V& value) : name_(name), value_(value) {}
  const char* name_;
  V& value_;
};
template <arg_print_way Way, typename V>
struct _v_argument_t {
  _v_argument_t(const char* name, const V& value)
      : name_(name), value_(value) {}
  const char* name_;
  V value_;
};

template <typename V>
using argument_t = _argument_t<print_direct, V>;

template <typename V>
using q_argument_t = _argument_t<print_quote, V>;

template <typename V>
using d_argument_t = _argument_t<print_deref, V>;

template <typename V>
using qd_argument_t = _argument_t<print_quote_deref, V>;

template <typename V>
using xd_argument_t = _argument_t<print_hex_deref, V>;

template <typename V>
using a_argument_t = _v_argument_t<print_array, V>;

template <typename V>
using x_argument_t = _argument_t<print_hex, V>;

template <typename V>
using xa_argument_t = _v_argument_t<print_hex_array, V>;

template <typename V>
using qa_argument_t = _v_argument_t<print_quote_array, V>;

// TODO: Remove this hack when new HCL_Request definition is ready.
//       The following code is a hack to easen the process of changing the
//       definition of symbol HCL_Request.
namespace hack {
template <typename ObjectT>
struct FormattableObject {
  ObjectT obj;

  friend std::ostream& operator<<(
      std::ostream& out,
      const FormattableObject<ObjectT>& fobj) {
    return fobj.Format(out);
  }

 private:
  std::ostream& Format(std::ostream& out) const {
    return out << obj;
  }
};

template <typename ObjectT>
FormattableObject<ObjectT> MakeFormattableObject(ObjectT obj) {
  return {std::move(obj)};
}
} // namespace hack

template <typename T>
struct FormattableAsHex {
  T t;
};

template <typename T>
inline std::ostream& operator<<(
    std::ostream& os,
    const FormattableAsHex<T>& v) {
  os << "\"0x" << std::hex << hack::MakeFormattableObject(v.t) << std::dec
     << '"';
  return os;
}

template <typename V>
inline std::ostream& operator<<(std::ostream& out, argument_t<V>&& v) {
  return out << '"' << v.name_ << "\":" << v.value_;
}

template <typename V>
inline std::ostream& operator<<(std::ostream& out, q_argument_t<V>&& v) {
  return out << '"' << v.name_ << "\":\"" << v.value_ << '"';
}

inline std::ostream& operator<<(
    std::ostream& out,
    q_argument_t<const char*>&& v) {
  return out << '"' << v.name_ << "\":\"" << (v.value_ ? v.value_ : "nullptr")
             << '"';
}

template <typename V>
inline std::ostream& operator<<(std::ostream& out, d_argument_t<V>&& v) {
  return out << '"' << v.name_ << "\":" << *v.value_;
}

template <typename V>
inline std::ostream& operator<<(std::ostream& out, qd_argument_t<V>&& v) {
  return out << '"' << v.name_ << "\":\"" << *v.value_ << '"';
}

template <typename V>
inline std::ostream& operator<<(std::ostream& out, xd_argument_t<V>&& v) {
  return out << '"' << v.name_
             << "\":" << FormattableAsHex<decltype(*v.value_)>{*v.value_};
}

template <typename V>
inline std::ostream& operator<<(std::ostream& out, x_argument_t<V>&& v) {
  return out << '"' << v.name_
             << "\":" << FormattableAsHex<decltype(v.value_)>{v.value_};
}

template <typename V>
inline std::ostream& operator<<(std::ostream& out, a_argument_t<V>&& v) {
  return out << '"' << v.name_ << "\":[" << v.value_ << ']';
}

template <typename V>
inline std::ostream& operator<<(
    std::ostream& out,
    qa_argument_t<absl::Span<V>>&& v) {
  return out << '"' << v.name_ << "\":\"" << v.value_ << '"';
}

template <typename V>
inline std::ostream& operator<<(
    std::ostream& out,
    xa_argument_t<absl::Span<V>>&& v) {
  out << '"' << v.name_ << "\":[";
  auto& buffer = v.value_;
  if (buffer.size() && buffer.begin()) {
    for (unsigned long i = 0; i < buffer.size() - 1; ++i) {
      out << FormattableAsHex<V>{buffer[i]} << ", ";
    }
    out << FormattableAsHex<V>{buffer[buffer.size() - 1]};
  }

  return out << ']';
}

template <typename Arg, typename... Args>
inline void concat_args(std::ostream& out, Arg&& arg, Args&&... args) {
  out << std::forward<Arg>(arg);
  using expander = int[]; // NOLINT
  (void)expander{0, (void(out << ',' << std::forward<Args>(args)), 0)...};
}

template <typename Arg>
inline void concat_args(std::ostream& out, Arg&& arg) {
  out << std::forward<Arg>(arg);
}

inline void concat_args([[maybe_unused]] std::ostream& out) {}

#define API_LOG_RESULT(...)                                                  \
  do {                                                                       \
    if (synapse_logger::log_observer_is_enabled(__FUNCTION__)) {             \
      synapse_logger::ostr_t out{synapse_logger::get_ostr()};                \
      out << "status:" << std::dec << status;                                \
      if (!synapse_logger::is_status_success((status))) {                    \
        out << R"(, "cname":"bad")";                                         \
      }                                                                      \
      comma_maybe(out, ##__VA_ARGS__);                                       \
      concat_args(out, ##__VA_ARGS__);                                       \
      synapse_logger::on_log(__FUNCTION__, out.str(), false);                \
    }                                                                        \
    if (!synapse_logger::logger_is_enabled(                                  \
            synapse_logger::data_dump_category::SYNAPSE_API_CALL)) {         \
      break;                                                                 \
    }                                                                        \
    synapse_logger::ostr_t out{synapse_logger::get_ostr()};                  \
    out << "\"name\":\"call:" << __FUNCTION__                                \
        << "\", \"ph\":\"E\", \"args\":{ \"status\":" << std::dec << status; \
    if (!synapse_logger::is_status_success((status))) {                      \
      out << R"(, "cname":"bad")";                                           \
    }                                                                        \
    comma_maybe(out, ##__VA_ARGS__);                                         \
    concat_args(out, ##__VA_ARGS__);                                         \
    out << "}";                                                              \
    synapse_logger::log(out.str());                                          \
  } while (false)

#define API_LOG_CALL(...)                                            \
  do {                                                               \
    if (synapse_logger::log_observer_is_enabled(__FUNCTION__)) {     \
      synapse_logger::ostr_t out{synapse_logger::get_ostr()};        \
      concat_args(out, ##__VA_ARGS__);                               \
      synapse_logger::on_log(__FUNCTION__, out.str(), true);         \
    }                                                                \
    if (!synapse_logger::logger_is_enabled(                          \
            synapse_logger::data_dump_category::SYNAPSE_API_CALL)) { \
      break;                                                         \
    }                                                                \
    synapse_logger::ostr_t out{synapse_logger::get_ostr()};          \
    out << "\"name\":\"call:" << __FUNCTION__                        \
        << "\", \"ph\":\"B\", \"func\":\"" << __PRETTY_FUNCTION__    \
        << "\", \"args\":{ ";                                        \
    concat_args(out, ##__VA_ARGS__);                                 \
    out << "}";                                                      \
    synapse_logger::log(out.str());                                  \
  } while (false)

// stringifiers to format args for json, explanations:
// suffix _X -> hex format
// suffix _Q -> quote
// prefix S_ -> pointer dereference versions
// prefix M_ -> multi
#define ARG(x)              \
  argument_t<decltype(x)> { \
#x, x                   \
  }
#define ARG_X(x)              \
  x_argument_t<decltype(x)> { \
#x, x                     \
  }
#define ARG_Q(x)              \
  q_argument_t<decltype(x)> { \
#x, x                     \
  }
#define S_ARG(x)              \
  d_argument_t<decltype(x)> { \
#x, x                     \
  }
#define S_ARG_Q(x)             \
  qd_argument_t<decltype(x)> { \
#x, x                      \
  }
#define S_ARG_X(x)             \
  xd_argument_t<decltype(x)> { \
#x, x                      \
  }
#define M_ARG(x, c)                              \
  a_argument_t<decltype(absl::MakeSpan(x, c))> { \
#x, absl::MakeSpan(x, c)                     \
  }
#define M_ARG_X(x, c)                             \
  xa_argument_t<decltype(absl::MakeSpan(x, c))> { \
#x, absl::MakeSpan(x, c)                      \
  }
#define M_ARG_Q(x, c)                             \
  qa_argument_t<decltype(absl::MakeSpan(x, c))> { \
#x, absl::MakeSpan(x, c)                      \
  }
