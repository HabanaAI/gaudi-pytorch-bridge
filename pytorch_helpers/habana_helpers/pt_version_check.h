/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
