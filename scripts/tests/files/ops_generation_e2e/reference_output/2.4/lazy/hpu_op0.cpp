// Autogenerated file by gen_op.py. Do not edit directly!

#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_validator.h"
#include "hpu_ops/op_logger.h"
#include "common/dump_args.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "hpu_ops/lazy/reduction_template.h"
#include "habana_lazy/hpu_stage_submission.h"
using habana_lazy::LazyOp;
using habana_lazy::GraphHashBuilder;

#include "__ilshift__.h"
#include "_foreach_add.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {



at::Tensor & __ilshift__(at::Tensor & self, const at::Scalar & other) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO("__ilshift__: ", DUMP_2ARGS(self, other));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{-1, {at::kInt, at::kChar, at::kByte, at::kShort, at::kBool}}}))
  FALLBACK_IF_UNSUPPORTED_DTYPE2(self, __ilshift__, Scalar, self, other)

  LazyOp<at::Tensor &> hpu_op{"aten::__ilshift__", {self, other}};
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(__ilshift__, hpu_op, self);
}

void _foreach_add_(at::TensorList self, const at::Scalar & scalar) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO("_foreach_add_: ", DUMP_2ARGS(self, scalar));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kLong, at::kInt, at::kShort, at::kChar, at::kDouble, at::kBool}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kLong, at::kInt, at::kShort, at::kChar, at::kHalf, at::kDouble, at::kBool}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kLong, at::kInt, at::kShort, at::kChar, at::kHalf, at::kDouble, at::kBool}}}))

  LazyOp<void> hpu_op{"aten::_foreach_add_", {self, scalar}};
  RUN_TENSOR_LIST_INPLACE_MAYBE_WITH_ACC_THREAD(_foreach_add_, hpu_op, self);
}





static const auto& kr_gen_0 = KernelRegistry()
;

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("__ilshift__.Scalar", static_cast<at::Tensor & (*)(at::Tensor &, const at::Scalar &)>(&habana::__ilshift__));
  m.impl("_foreach_add_.Scalar", static_cast<void (*)(at::TensorList, const at::Scalar &)>(&habana::_foreach_add_));

}



}  // namespace habana

