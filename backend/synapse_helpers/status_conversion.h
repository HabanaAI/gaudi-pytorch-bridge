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
