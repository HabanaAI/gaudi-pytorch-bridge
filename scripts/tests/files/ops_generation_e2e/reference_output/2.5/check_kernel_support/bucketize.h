// Autogenerated file by gen_op.py. Do not edit directly!

#pragma once
#include <ATen/Tensor.h>

#include "habana_eager/ops/eager_op.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {


at::Tensor bucketize(const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right);
OUTMETA_DECL(BucketizeMeta);
FILL_PARAMS_DECL(FillBucketizeParams);

}  // namespace habana

