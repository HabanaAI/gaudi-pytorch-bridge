// Autogenerated file by gen_op.py. Do not edit directly!

#pragma once
#include <ATen/Tensor.h>

#include "habana_eager/ops/eager_op.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {


at::Tensor as_strided(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<c10::SymInt> storage_offset);

}  // namespace habana

