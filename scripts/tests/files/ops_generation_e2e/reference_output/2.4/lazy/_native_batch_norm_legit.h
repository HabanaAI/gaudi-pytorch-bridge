// Autogenerated file by gen_op.py. Do not edit directly!

#pragma once
#include <ATen/Tensor.h>
#include <ATen/core/function_schema.h>
#include "habana_kernels/lazy_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {


::std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, bool training, double momentum, double eps);
OUTMETA_DECL(BatchNormFwdMeta);
FILL_PARAMS_DECL(FillBatchNormFwdParams);

}  // namespace habana

