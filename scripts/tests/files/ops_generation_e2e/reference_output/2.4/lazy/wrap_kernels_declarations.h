// Autogenerated file by gen_op.py. Do not edit directly!

#pragma once
#include <ATen/ExpandUtils.h>
#include <torch/script.h>
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace hpu_wrap {

at::Tensor matmul(const at::Tensor &, const at::Tensor &);
at::Tensor _reshape_alias(const at::Tensor &, c10::SymIntArrayRef, c10::SymIntArrayRef);
at::Tensor dropout(const at::Tensor &, double, bool);

} // namespace hpu_wrap

