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
