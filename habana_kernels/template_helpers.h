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

#include <ATen/Tensor.h>
#include <tuple>
#include <type_traits>

template <class...>
struct conjunction : std::true_type {};

template <class B1>
struct conjunction<B1> : B1 {};

template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

template <typename Tuple>
struct is_tuple_of_tensor_ref;

template <typename Tuple>
struct is_tuple_of_tensors;

template <typename... Ts>
struct is_tuple_of_tensor_ref<std::tuple<Ts...>>
    : conjunction<std::is_same<at::Tensor&, Ts>...> {};

template <typename... Ts>
struct is_tuple_of_tensors<std::tuple<Ts...>>
    : conjunction<std::is_same<at::Tensor, Ts>...> {};

namespace habana {

template <class F, class... Ts>
void for_each_in_tuple(std::tuple<Ts...>& tuple, F&& func) {
  std::apply([&func](auto&... args) { (func(args), ...); }, tuple);
}

template <class F, class... Ts>
void for_each_in_tuple_with_index(std::tuple<Ts...>& tuple, F&& func) {
  std::apply(
      [&func](auto&... args) {
        size_t index = 0;
        (func(args, index++), ...);
      },
      tuple);
}

} // namespace habana
