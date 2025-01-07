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
#include "hpu_ops/op_backend.h"
#include "include/habanalabs/hpu_custom_op_pt2.h"

namespace habana {

// A class that represents user's torch custom op
// All user's torch ops with a single user tpc kernel will pass here
// when lowered to synapse graph.
class UserCustomOpBackend : public OpBackend {
 public:
  UserCustomOpBackend(
      int device_id,
      const habana::custom_op::UserCustomOpDescriptor& desc)
      : OpBackend(
            device_id,
            desc.getGuid(),
            c10::ScalarType::Undefined, // This kernel will be called without
                                        // dtype suffix.
            {0},
            {},
            {},
            false) {
    SetOutputMetaFn(desc.getOutputMetaFn());
    SetFillParams(desc.getFillParamsFn());
  }
};

} // namespace habana
