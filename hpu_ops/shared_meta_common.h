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

#include "hpu_ops/hpu_op_helper.h"

#define SHARED_META(name)                \
  SharedMetaDataVector name##SharedMeta( \
      const at::Stack& stack,            \
      habana_helpers::HabanaExecutionMode executionMode);

#define SHARED_META_GUID_EXEC_MODE(name) \
  SharedMetaDataVector name##SharedMeta( \
      const at::Stack& stack,            \
      const std::string& guid,           \
      habana_helpers::HabanaExecutionMode executionMode);

#define SHARED_META_GUID(name)           \
  SharedMetaDataVector name##SharedMeta( \
      const at::Stack& stack, const std::string& guid);

namespace habana {

SHARED_META_GUID(Input0)
SHARED_META_GUID(Input0ToOut0And1)
SHARED_META_GUID(AdaptiveBwd)
SHARED_META_GUID(AvgPoolBwd)
SHARED_META_GUID(FillCumSumProd)
SHARED_META_GUID(IsFiniteInfNan)
SHARED_META_GUID(Rounding)
SHARED_META_GUID(Compare)
SHARED_META_GUID(ForeachCompound)
SHARED_META(BoolCast)
SHARED_META_GUID(LogicalBinary)
SHARED_META_GUID(UnaryForeach)
SHARED_META_GUID_EXEC_MODE(AminAmax)
SHARED_META_GUID(BinaryWithAlpha)
SHARED_META_GUID(BitwiseLogical)
SHARED_META(Topk)
SHARED_META_GUID(RandomSeedTensorInput)
SHARED_META_GUID(MaxPoolWithIndicesFwd)
SHARED_META_GUID(MaxPoolWithIndicesBwd)
SHARED_META(Empty)
SHARED_META(Matmul)
SHARED_META(StridedView)

} // namespace habana
