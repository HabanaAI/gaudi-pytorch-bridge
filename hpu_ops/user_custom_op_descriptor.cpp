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
#include "hpu_ops/user_custom_op_backend.h"
#include "include/habanalabs/hpu_custom_op_pt2.h"

namespace habana {
namespace custom_op {

const UserCustomOpDescriptor& UserCustomOpDescriptor::getUserCustomOpDescriptor(
    const std::string& op) {
  return habana::KernelRegistry().get_user_custom_op_desc(op);
}

const std::string& UserCustomOpDescriptor::getSchemaName() const {
  return schema_;
}

const std::string& UserCustomOpDescriptor::getGuid() const {
  return guid_;
}

const FillParamsFn& UserCustomOpDescriptor::getFillParamsFn() const {
  return fill_params_fn_;
}

const OutputMetaFn& UserCustomOpDescriptor::getOutputMetaFn() const {
  return output_meta_fn_;
}

namespace {
void registerKernel(const UserCustomOpDescriptor& new_desc) {
  habana::KernelRegistry().add_user_custom_op(
      [&](const int device_id, const std::string& schema_name) {
        auto& desc =
            habana::KernelRegistry().get_user_custom_op_desc(schema_name);
        return std::make_shared<habana::UserCustomOpBackend>(device_id, desc);
      },
      new_desc);
}
} // namespace

void registerUserCustomOp(
    const std::string& schema,
    const std::string& guid,
    OutputMetaFn output_meta_fn,
    FillParamsFn fill_params_fn) {
  UserCustomOpDescriptor op_desc{schema, guid, output_meta_fn, fill_params_fn};
  registerKernel(op_desc);
}

} // namespace custom_op
} // namespace habana
