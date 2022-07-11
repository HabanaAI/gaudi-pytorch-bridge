/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
