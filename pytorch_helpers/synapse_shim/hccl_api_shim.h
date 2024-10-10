/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************/
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
