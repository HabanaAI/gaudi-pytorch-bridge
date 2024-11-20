/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/
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
