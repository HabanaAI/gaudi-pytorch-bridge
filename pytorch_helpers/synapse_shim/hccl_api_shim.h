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

#include <functional>

#include <hccl.h> // IWYU pragma: keep
#include "hccl_types.h"

#define HCCL_API_SYMBOL_VISIT(visitor) \
  visitor(hcclGetVersion);             \
  visitor(hcclGetUniqueId);            \
  visitor(hcclCommInitRank);           \
  visitor(hcclCommInitAll);            \
  visitor(hcclCommDestroy);            \
  visitor(hcclCommAbort);              \
  visitor(hcclGetErrorString);         \
  visitor(hcclCommGetAsyncError);      \
  visitor(hcclCommCount);              \
  visitor(hcclCommUserRank);           \
  visitor(hcclLookupDMABuff);          \
  visitor(hcclReduce);                 \
  visitor(hcclBroadcast);              \
  visitor(hcclAllReduce);              \
  visitor(hcclReduceScatter);          \
  visitor(hcclAllGather);              \
  visitor(hcclAlltoAll);               \
  visitor(hcclSend);                   \
  visitor(hcclRecv);                   \
  visitor(hcclGroupStart);             \
  visitor(hcclGroupEnd);

#define DECL_HCCL_FN(func)                      \
  using func##_pfn_t = decltype(::func);        \
  using func##_t = std::function<func##_pfn_t>; \
  func##_t func{};

struct hccl_api_t {
  HCCL_API_SYMBOL_VISIT(DECL_HCCL_FN);
};

extern hccl_api_t* hccl_api;
hccl_api_t* GetHcclApi();
