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

#pragma once

#include "hccl.h"
#include "synapse_error.h"

namespace hccl_integration {

hcclResult_t to_hccl_result(const synapse_helpers::synapse_error& status);
hcclResult_t to_hccl_result(synStatus status);

} // namespace hccl_integration

#define RETURN_ON_SYNAPSE_ERROR(status)                  \
  {                                                      \
    hcclResult_t hccl_status = to_hccl_result((status)); \
    if (hcclSuccess != hccl_status) {                    \
      return hccl_status;                                \
    }                                                    \
  }
