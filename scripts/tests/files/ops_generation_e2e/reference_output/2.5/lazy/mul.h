// Autogenerated file by gen_op.py. Do not edit directly!

#pragma once
#include <ATen/Tensor.h>
#include <ATen/core/function_schema.h>
#include "habana_kernels/lazy_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {


at::Tensor & mul_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out);
OUTSHAPE_DECL(BinaryOutputShape);

}  // namespace habana

