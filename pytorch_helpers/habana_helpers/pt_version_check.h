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

#include <torch/csrc/api/include/torch/version.h>

#ifdef PYTORCH_FORK
#define IS_FORK 1
#else
#define IS_FORK 0
#endif

#define IS_PYTORCH_FORK_AT_LEAST(MAJOR, MINOR) \
  IS_FORK == 1 &&                              \
      (PYTORCH_FORK_MAJOR > MAJOR ||           \
       (PYTORCH_FORK_MAJOR == MAJOR && PYTORCH_FORK_MINOR >= MINOR))

#define IS_PYTORCH_AT_LEAST(MAJOR, MINOR) \
  (TORCH_VERSION_MAJOR > MAJOR ||         \
   (TORCH_VERSION_MAJOR == MAJOR && TORCH_VERSION_MINOR >= MINOR))

#define IS_PYTORCH_OLDER_THAN(MAJOR, MINOR) \
  (TORCH_VERSION_MAJOR < MAJOR ||           \
   (TORCH_VERSION_MAJOR == MAJOR && TORCH_VERSION_MINOR < MINOR))

#define IS_PYTORCH_EXACTLY(MAJOR, MINOR) \
  (TORCH_VERSION_MAJOR == MAJOR && TORCH_VERSION_MINOR == MINOR)
