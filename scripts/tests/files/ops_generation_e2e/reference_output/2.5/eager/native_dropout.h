// Autogenerated file by gen_op.py. Do not edit directly!

#pragma once
#include <ATen/Tensor.h>

#include "habana_eager/ops/eager_op.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

HPU_OP_FRONTEND(eager::EagerOp, NativeDropoutFE)

::std::tuple<at::Tensor,at::Tensor> native_dropout(const at::Tensor & input, double p, c10::optional<bool> train);
unsigned NativeDropoutEarlyExitCondition(const at::Tensor & input, double p, c10::optional<bool> train);
::std::tuple<at::Tensor,at::Tensor> NativeDropoutEarlyExit(unsigned eePath, const at::Tensor & input, double p, c10::optional<bool> train);
OUTMETA_DECL(FusedNativeDropoutMeta);
FILL_PARAMS_DECL(FillFusedNativeDropoutParams);

}  // namespace habana

