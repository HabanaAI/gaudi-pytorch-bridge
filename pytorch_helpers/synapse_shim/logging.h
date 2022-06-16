/*******************************************************************************
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
