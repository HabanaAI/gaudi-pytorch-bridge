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
#include <iostream>
#include <string>

namespace habana {
enum class LayoutFormat { NHWC = 0, NCHW = 1, HWCK = 2, ANY = 3, INVALID = 4 };

class LayoutFormatDims {
 public:
  constexpr static char N = 0;
  constexpr static char C = 1;
  constexpr static char H = 2;
  constexpr static char W = 3;
};

class LayoutFormatWithDepthDims {
 public:
  constexpr static char N = 0;
  constexpr static char C = 1;
  constexpr static char D = 2;
  constexpr static char H = 3;
  constexpr static char W = 4;
};

inline std::string DebugString(const LayoutFormat& l) {
  switch (l) {
    case LayoutFormat::NHWC:
      return std::string("NHWC");
    case LayoutFormat::NCHW:
      return std::string("NCHW");
    case LayoutFormat::HWCK:
      return std::string("HWCK");
    case LayoutFormat::ANY:
      return std::string("ANY");
    default:
      return std::string("kINVALID");
  }
  return std::string();
}

inline std::ostream& operator<<(std::ostream& O, const LayoutFormat& l) {
  return O << DebugString(l);
}
} // namespace habana
