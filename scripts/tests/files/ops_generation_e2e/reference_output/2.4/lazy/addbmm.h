// Autogenerated file by gen_op.py. Do not edit directly!

#pragma once
#include <ATen/Tensor.h>
#include <ATen/core/function_schema.h>
#include "habana_kernels/lazy_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {


at::Tensor addbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha);
OUTMETA_DECL(AddBMMMeta);
SHARED_LAYER_META_DECL(AddBMMSharedMeta);

}  // namespace habana

