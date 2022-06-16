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

#define CHECK_NULL(x)                                                       \
  do {                                                                      \
    if (nullptr == (x)) {                                                   \
      std::cerr << "ERROR: pid = " << getpid() << " at " << __FILE__ << ":" \
                << __LINE__ << " (" << dlerror() << ")\n";                  \
      std::terminate();                                                     \
    }                                                                       \
  } while (0)

#define CHECK_TRUE(x)                                                       \
  do {                                                                      \
    if (!(x)) {                                                             \
      std::cerr << "ERROR: pid = " << getpid() << " at " << __FILE__ << ":" \
                << __LINE__ << " (" << dlerror() << ")\n";                  \
      std::terminate();                                                     \
    }                                                                       \
  } while (0)

} // namespace shim
