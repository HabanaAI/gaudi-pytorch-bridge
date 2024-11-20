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
