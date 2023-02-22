
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
#include "habana_helpers/logging.h"

#define EAGER_NOT_SUPPORTED                                                   \
  HABANA_ASSERT(                                                              \
      false, "Frontend Op ", __func__, " not supported with new Eager mode"); \
  std::terminate();
