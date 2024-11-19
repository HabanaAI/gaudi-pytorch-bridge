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

// Allows looping over std::get<i>(v)
// Instead of:
//    std::get<0>(v)
//    std::get<1>(v)
//    ...
//    std::get<N-1>(v)
// You can write a loop:
//    for i = 0 to N-1:
//      VariableGet<0, N>::get(i, v)
template <int I, int N>
struct VariableGet {
  template <class T>
  static auto get(int i, const T& v) {
    if (i == I) {
      return std::get<I>(v);
    } else {
      return VariableGet<I + 1, N>::get(i, v);
    }
  }
};

template <int N>
struct VariableGet<N, N> {
  template <class T>
  static auto get(int, const T& v) {
    return decltype(std::get<0>(v)){};
  }
};
