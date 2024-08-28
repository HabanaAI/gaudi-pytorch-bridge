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
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"

static habana::PartialOutputMetaDataVector output_meta(
    const at::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto other = inputs[1].toTensor();
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  habana::PartialOutputMetaData meta_output{self.scalar_type(), output_shape};
  return {meta_output};
}

bool register_custom_add() {
  habana::custom_op::registerUserCustomOp(
      "custom_op::custom_add", "add_fwd_f32", output_meta, nullptr);
  return true;
}

at::Tensor custom_add(at::Tensor input_a, at::Tensor input_b) {
  std::vector<c10::IValue> inputs{input_a, input_b};
  auto op_desc =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          "custom_op::custom_add");
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  return output[0];
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("custom_add(Tensor self, Tensor other) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_add", custom_add);
}

at::Tensor custom_add_meta(at::Tensor input_a, at::Tensor input_b) {
  auto output_shape = at::infer_size(input_a.sizes(), input_b.sizes());
  return input_a.new_empty(output_shape, input_a.scalar_type());
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("custom_add", &custom_add_meta);
}

static const auto& KernelReg = register_custom_add();
