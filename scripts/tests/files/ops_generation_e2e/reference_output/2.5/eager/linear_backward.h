// Autogenerated file by gen_op.py. Do not edit directly!

#pragma once
#include <ATen/Tensor.h>

#include "habana_eager/ops/eager_op.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {


::std::tuple<at::Tensor,at::Tensor,at::Tensor> linear_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, ::std::array<bool,3> output_mask);
OUTMETA_DECL(LinearBackwardMeta);
FILL_PARAMS_DECL(FillLinearBwdParams);

}  // namespace habana

