/**
 * Copyright (c) 2025 Intel Corporation
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

// clang-format off

#define SUPPRESS_W_PREFIX                                         \
  _Pragma("GCC diagnostic push")                                  \
  _Pragma("GCC diagnostic ignored \"-Wpragmas\"")                 \
  _Pragma("GCC diagnostic ignored \"-Wunknown-warning-option\"")

#define SUPPRESS_W_SUFFIX _Pragma("GCC diagnostic pop")

#define SUPPRESS_W_TEMPLATE(...)                                  \
  SUPPRESS_W_PREFIX                                               \
  __VA_ARGS__                                                     \
  SUPPRESS_W_SUFFIX

// GCC 13 has a bug with -Wdangling-reference
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107532
#define SUPPRESS_WDANGLING_REFERENCE_P                            \
  _Pragma("GCC diagnostic ignored \"-Wdangling-reference\"")

#define SUPPRESS_WDANGLING_REFERENCE(...)                         \
  SUPPRESS_W_TEMPLATE(                                            \
  SUPPRESS_WDANGLING_REFERENCE_P                                  \
  __VA_ARGS__                                                     \
  )

// GCC 13 has a bug with -Warray-bounds and -Wstringop-overflow
// affecting std::vector::reserve
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=110498
#define SUPPRESS_WARRAY_BOUNDS_WSTRINGOP_OVERFLOW_P               \
  _Pragma("GCC diagnostic ignored \"-Warray-bounds\"")            \
  _Pragma("GCC diagnostic ignored \"-Wstringop-overflow\"")

#define SUPPRESS_WARRAY_BOUNDS_WSTRINGOP_OVERFLOW(...)            \
  SUPPRESS_W_TEMPLATE(                                            \
  SUPPRESS_WARRAY_BOUNDS_WSTRINGOP_OVERFLOW_P                     \
  __VA_ARGS__                                                     \
  )

// GCC is dumb and always chooses operator* over operator bool,
// no matter how user tries to override this. But this triggers
// Wconversion with no other option than suppress.
#define SUPPRESS_WCONVERSION_P                                    \
  _Pragma("GCC diagnostic ignored \"-Wconversion\"")

#define SUPPRESS_WCONVERSION(...)                                 \
  SUPPRESS_W_TEMPLATE(                                            \
  SUPPRESS_WCONVERSION_P                                          \
  __VA_ARGS__                                                     \
  )

// clang-format on
