/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

namespace habana {

inline int64_t mod_exp(int64_t y, int64_t x = 997) {
  const int64_t p{1000000007};
  int64_t z = 1;
  y = y % p;
  if (y == 0) {
    return 0;
  }

  while (y > 0) {
    if (y & 1) {
      z = (z * x) % p;
    }

    y >>= 1;
    x = (x * x) % p;
  }
  return z;
}

inline int64_t mod_exp(bool w, int64_t x = 997) {
  int64_t y = (w ? 97 : 43);
  return (mod_exp(y, x));
}

} // namespace habana
