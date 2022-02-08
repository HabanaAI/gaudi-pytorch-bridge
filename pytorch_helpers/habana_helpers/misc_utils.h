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

#include <cstdlib>

#include <iostream>
#include <string>

#include <ATen/Tensor.h>

namespace habana {

// Returns if the input is a ZST
inline bool is_ZST(const at::Tensor& t) {
  // NOTE: There are cases where the numel returns 0 but the tensor has non-null
  // storage pointer. REPRODUCER: LazyDynamicShapesTest.ArangeTest
  return (
      (t.has_storage() && t.storage().data_ptr().get() == nullptr) ||
      (t.numel() == 0));
}

// Computes (x^y)%1000000007
inline int64_t mod_exp(int64_t y, int64_t x = 997) {
  const int64_t p{1000000007};
  int64_t z = 1;
  int64_t sign{(y < 0 ? -1 : 1)};
  y = llabs(y);

  x = x % p;
  if (x == 0) {
    return 0;
  }

  while (y > 0) {
    if (y & 1) {
      z = (z * x) % p;
    }

    y >>= 1;
    x = (x * x) % p;
  }
  z *= sign;
  return z;
}

inline int64_t mod_exp(bool w, int64_t x = 997) {
  int64_t y = (w ? 97 : 43);
  return (mod_exp(y, x));
}

} // namespace habana

namespace habana_lazy {

enum LayoutFormat { kNHWC = 0, kNCHW = 1, kHWCK = 2, kANY = 3, kINVALID = 4 };

inline std::string DebugString(const LayoutFormat& l) {
  switch (l) {
    case LayoutFormat::kNHWC:
      return std::string("NHWC");
    case LayoutFormat::kNCHW:
      return std::string("NCHW");
    case LayoutFormat::kHWCK:
      return std::string("HWCK");
    case LayoutFormat::kANY:
      return std::string("kANY");
    default:
      return std::string("kINVALID");
  }
  return std::string();
}

inline std::ostream& operator<<(std::ostream& O, const LayoutFormat& l) {
  return O << DebugString(l);
}

bool IsCollective(const c10::Symbol& symbol);

} // namespace habana_lazy
