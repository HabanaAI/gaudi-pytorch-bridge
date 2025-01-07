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
#include "hpu_ops/op_backend.h"

namespace habana {

struct Int4BaseOp : OpBackend {
  Int4BaseOp(
      int device_id,
      c10::ScalarType scalar_type,
      const std::string& guid);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

#define DEFINE_OP(op)                               \
  struct op : Int4BaseOp {                          \
    op(int device_id, c10::ScalarType scalar_type); \
  };

DEFINE_OP(ConvertFromInt4)
DEFINE_OP(ConvertFromUint4)

OUTMETA_DECL(ConvertFromInt4Meta)

} // namespace habana