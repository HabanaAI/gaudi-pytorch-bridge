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
#include <unistd.h>
#include <array>
#include <iostream>

namespace shim {

#define CHECK_NULL(x) CHECK_NULL_MSG(x, "")

#define CHECK_NULL_MSG(x, msg) CHECK_TRUE_MSG(nullptr != (x), msg)

#define CHECK_TRUE(x) CHECK_TRUE_MSG(x, "")
#define CHECK_TRUE_DL(x) CHECK_TRUE_MSG(x, " (" << dlerror() << ")")

#define CHECK_TRUE_MSG(x, msg)                                              \
  do {                                                                      \
    if (!(x)) {                                                             \
      std::cerr << "ERROR: pid = " << getpid() << " at " << __FILE__ << ":" \
                << __LINE__ << " " << msg << std::endl;                     \
      std::terminate();                                                     \
    }                                                                       \
  } while (0)

} // namespace shim
