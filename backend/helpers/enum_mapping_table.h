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

#include <array>

template <class E, class T>
struct EnumMappingTable {
  T& operator[](E e) {
    return v_[static_cast<size_t>(e)];
  }
  const T& operator[](E e) const {
    return v_[static_cast<size_t>(e)];
  }

  std::array<T, static_cast<size_t>(E::__count)> v_;
};
