/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
