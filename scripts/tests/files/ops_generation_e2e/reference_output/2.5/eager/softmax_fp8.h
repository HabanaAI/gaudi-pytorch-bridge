// Autogenerated file by gen_op.py. Do not edit directly!

#pragma once
#include <ATen/Tensor.h>

#include "habana_eager/ops/eager_op.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {


at::Tensor softmax_fp8(const at::Tensor & input, int64_t dim, const c10::optional<at::Tensor> & input_scale, const c10::optional<at::Tensor> & output_scale, const c10::optional<at::Tensor> & inv_attn_heads, const c10::optional<at::Tensor> & fused_add);
OUTMETA_DECL(SoftmaxFp8Meta);

}  // namespace habana

