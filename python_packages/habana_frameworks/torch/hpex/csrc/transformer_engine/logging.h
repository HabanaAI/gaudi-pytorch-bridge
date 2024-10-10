/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/
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
 * Changes:
 * - Modified naming of NVTE_ERROR and NVTE_CHECK to HPTE_ERROR and HPTE_CHECK
 * - Removed unused macros
 ******************************************************************************/

#ifndef TRANSFORMER_ENGINE_LOGGING_H_
#define TRANSFORMER_ENGINE_LOGGING_H_

#include <stdexcept>
#include <string>

#define HPTE_ERROR(x)                                          \
  do {                                                         \
    throw std::runtime_error(                                  \
        std::string(__FILE__ ":") + std::to_string(__LINE__) + \
        " in function " + __func__ + ": " + x);                \
  } while (false)

#define HPTE_CHECK(x, ...)                            \
  do {                                                \
    if (!(x)) {                                       \
      HPTE_ERROR(                                     \
          std::string("Assertion failed: " #x ". ") + \
          std::string(__VA_ARGS__));                  \
    }                                                 \
  } while (false)

#endif // TRANSFORMER_ENGINE_LOGGING_H_
