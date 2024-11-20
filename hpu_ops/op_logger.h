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

// clang-format off
// PT 1.12 requires ArrayRef.h to come before IListRef.h because of implicit
// dependency of the latter on the former in PT sources
#include <c10/util/ArrayRef.h>
// clang-format on
#include <ATen/core/IListRef.h>
#include <ATen/core/TensorBody.h>
#include <c10/util/OptionalArrayRef.h>
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {

template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& out, const std::array<T, N>& seq) {
  c10::PrintSequence(out, seq.begin(), seq.end());
  return out;
}

template <typename T>
std::string to_string(const T& val) {
  std::ostringstream oss;
  oss << std::fixed << std::boolalpha << val;
  return oss.str();
}

template <typename T>
std::string to_string(const c10::optional<T>& val) {
  return val.has_value() ? to_string(*val) : "None";
}

template <typename T>
std::string to_string(const c10::OptionalArrayRef<T>& val) {
  return val.has_value() ? to_string(*val) : "None";
}

template <>
std::string to_string(const at::Tensor& t);

template <>
std::string to_string(const at::Generator&);

template <class ElementType, class Iter>
inline void print_list(std::ostringstream& out, Iter begin, Iter end) {
  // Output at most 100 elements -- appropriate if used for logging.
  for (int i = 0; begin != end && i < 100; ++i, ++begin) {
    if (i > 0)
      out << ' ';
    out << to_string(static_cast<ElementType>(*begin));
  }
  if (begin != end) {
    out << " ...";
  }
}

#define INSTANTIATE_FOR_LIST(list_type, element_type)                 \
  template <>                                                         \
  inline std::string to_string(const list_type<element_type>& list) { \
    std::ostringstream oss;                                           \
    print_list<element_type>(oss, list.begin(), list.end());          \
    return oss.str();                                                 \
  }

INSTANTIATE_FOR_LIST(c10::ArrayRef, at::Tensor)
INSTANTIATE_FOR_LIST(c10::IListRef, at::Tensor)
INSTANTIATE_FOR_LIST(c10::List, at::Tensor)
INSTANTIATE_FOR_LIST(c10::IListRef, at::OptionalTensorRef)
INSTANTIATE_FOR_LIST(c10::List, c10::optional<at::Tensor>)
INSTANTIATE_FOR_LIST(std::vector, at::Tensor)
#undef INSTANTIATE_FOR_LIST
} // namespace habana
