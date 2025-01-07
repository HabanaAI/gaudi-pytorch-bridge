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
