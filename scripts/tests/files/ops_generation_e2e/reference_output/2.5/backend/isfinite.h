// Autogenerated file by gen_op.py. Do not edit directly!

#pragma once
#include <ATen/Tensor.h>
#include <ATen/core/function_schema.h>
#include "habana_kernels/lazy_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {
HPU_OP_BACKEND(_IsFiniteInfNan)


at::Tensor isfinite(const at::Tensor & self);

}  // namespace habana

