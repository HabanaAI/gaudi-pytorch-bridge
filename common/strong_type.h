/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include <fmt/format.h>
#include <functional>
#include <type_traits>
#include <utility>

namespace common {

template <typename T>
using equality_comparable_t = decltype(
    std::declval<std::remove_reference_t<T>&>() ==
    std::declval<std::remove_reference_t<T>&>());

template <typename T, typename = std::void_t<>>
struct is_equality_comparable : std::false_type {};

template <typename T>
struct is_equality_comparable<T, std::void_t<equality_comparable_t<T>>>
    : std::is_same<equality_comparable_t<T>, bool>::type {};

/**
Allows descriptive passing of parameters of basic types

Disallows implicit conversions.

For example:
using TensorId = StrongType<int, struct TensorIdTag>;
using DimsCount = StrongType<int, struct DimsCountTag>;

MakeTensor(5, 8) - would not compile (no implicit conversion)
                   and it is not clear what are these numbers

MakeTensor(TensorId(5), DimsCount(8)) - self descriptive
*/
template <class T, class Tag, class CRTP>
class StrongTypeBase {
 public:
  using TnoRef = std::remove_reference_t<T>;

  const TnoRef& operator*() const {
    return get_value();
  }

  template <class T2 = T>
  std::enable_if_t<!std::is_const_v<T2>, TnoRef&> operator*() {
    return get_value();
  }

  std::remove_pointer_t<T>* operator->() {
    if constexpr (std::is_pointer_v<T>) {
      return get_value();
    } else {
      return &get_value();
    }
  }

  template <
      class T2 = T,
      std::enable_if_t<std::is_convertible_v<T2, bool>, bool> = 0>
  explicit operator bool() const {
    return static_cast<bool>(get_value());
  }

  template <
      class T2 = T,
      typename Enable = std::enable_if_t<is_equality_comparable<T2>::value>>
  bool operator==(const StrongTypeBase<T, Tag, CRTP>& rhs) const {
    return get_value() == *rhs;
  }

  template <
      class T2 = T,
      typename Enable = std::enable_if_t<is_equality_comparable<T2>::value>>
  bool operator!=(const StrongTypeBase<T, Tag, CRTP>& rhs) const {
    return !(get_value() == *rhs);
  }

  template <class T2, class Tag2, class CRTP2, typename Enable>
  friend std::ostream& operator<<(
      std::ostream& os,
      const StrongTypeBase<T2, Tag2, CRTP2>& v);

 private:
  TnoRef& get_value() {
    return static_cast<CRTP*>(this)->get_value_p();
  }

  const TnoRef& get_value() const {
    return static_cast<const CRTP*>(this)->get_value_p();
  }
};

template <class T, class Tag, typename Reference = void>
class StrongType : public StrongTypeBase<T, Tag, StrongType<T, Tag>> {
 public:
  explicit StrongType(T in)
      : StrongTypeBase<T, Tag, StrongType<T, Tag>>{}, value{std::move(in)} {}

  friend class StrongTypeBase<T, Tag, StrongType<T, Tag>>;

 private:
  T value;
  T& get_value_p() {
    return value;
  }

  const T& get_value_p() const {
    return value;
  }
};

template <class T, class Tag>
class StrongType<T, Tag, std::enable_if_t<std::is_reference_v<T>>>
    : public StrongTypeBase<T, Tag, StrongType<T, Tag>> {
 public:
  explicit StrongType(T in)
      : StrongTypeBase<T, Tag, StrongType<T, Tag>>{}, value{in} {}

  friend class StrongTypeBase<T, Tag, StrongType<T, Tag>>;

 private:
  std::reference_wrapper<std::remove_reference_t<T>> value;

  std::remove_reference_t<T>& get_value_p() {
    return value.get();
  }

  const std::remove_reference_t<T>& get_value_p() const {
    return value.get();
  }
};

template <typename T>
using ostream_operator_capable_t = std::remove_reference_t<decltype(
    std::declval<std::ostream>()
    << std::declval<std::remove_reference_t<T>>())>;

template <typename T, typename = std::void_t<>>
struct is_ostream_operator_capable : std::false_type {};

template <typename T>
struct is_ostream_operator_capable<
    T,
    std::void_t<ostream_operator_capable_t<T>>>
    : std::is_same<ostream_operator_capable_t<T>, std::ostream>::type {};
} // namespace common

template <
    class T,
    class Tag,
    class CRTP,
    typename Enable =
        std::enable_if_t<common::is_ostream_operator_capable<T>::value>>
std::ostream& operator<<(
    std::ostream& os,
    const common::StrongTypeBase<T, Tag, CRTP>& v) {
  os << (*v);
  return os;
}

namespace fmt {
template <class T, class Tag>
struct formatter<
    common::StrongType<T, Tag>,
    char,
    std::enable_if_t<fmt::is_formattable<std::remove_reference_t<T>>::value>> {
  formatter<std::remove_reference_t<T>> formatter;

  constexpr auto parse(format_parse_context& ctx) {
    return formatter.parse(ctx);
  }

  template <typename FormatContext>
  auto format(const common::StrongType<T, Tag>& v, FormatContext& ctx) const {
    return formatter.format(*v, ctx);
  }
};
} // namespace fmt

namespace common {
template <typename First, typename... Others>
using first_t = First;

template <typename T, typename = std::void_t<>>
struct has_std_hash : std::false_type {};

template <typename T>
struct has_std_hash<
    T,
    std::void_t<decltype(std::declval<std::hash<T>>()(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
constexpr bool has_std_hash_v = has_std_hash<T>::value;

} // namespace common

namespace std {
template <class T, class Tag>
struct hash<common::first_t<
    common::StrongType<T, Tag>,
    std::enable_if_t<common::has_std_hash_v<std::remove_reference_t<T>>>>> {
  std::uint64_t operator()(const common::StrongType<T, Tag>& v) const {
    return std::hash<std::remove_reference_t<T>>()(*v);
  }
};

} // namespace std