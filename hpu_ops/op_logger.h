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
 *******************************************************************************
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
