// Autogenerated file by gen_op.py. Do not edit directly!

#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_validator.h"
#include "hpu_ops/op_logger.h"
#include "common/dump_args.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/hpu_stage_submission.h"
using habana_lazy::LazyOp;
using habana_lazy::GraphHashBuilder;

#include "linear_backward.h"
#include "native_group_norm.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {



::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, double eps) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO("native_group_norm: ", DUMP_8ARGS(input, weight, bias, N, C, HxW, group, eps));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kDouble}}}))
  FALLBACK_IF_UNSUPPORTED_DTYPE(input, native_group_norm, input, weight, bias, N, C, HxW, group, eps)

  LazyOp<::std::tuple<at::Tensor,at::Tensor,at::Tensor>> hpu_op{"aten::native_group_norm", {input, weight, bias, N, C, HxW, group, eps}};
  hpu_op.SetOutputMetaFn(GroupNormFwdMeta);
  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(native_group_norm, hpu_op);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> linear_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, ::std::array<bool,3> output_mask) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO("linear_backward: ", DUMP_4ARGS(self, grad_output, weight, output_mask));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}}}))
  FALLBACK_IF_UNSUPPORTED_DTYPE(self, linear_backward, self, grad_output, weight, output_mask)
  FALLBACK_IF_UNSUPPORTED_DTYPE(grad_output, linear_backward, self, grad_output, weight, output_mask)
  FALLBACK_IF_UNSUPPORTED_DTYPE(weight, linear_backward, self, grad_output, weight, output_mask)

  LazyOp<::std::tuple<at::Tensor,at::Tensor,at::Tensor>> hpu_op{"aten::linear_backward", {self, grad_output, weight, output_mask}};
  hpu_op.SetOutputMetaFn(LinearBackwardMeta);
  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(linear_backward, hpu_op);
}





static const auto& kr_gen_9 = KernelRegistry()
;

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("native_group_norm", static_cast<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (*)(const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, c10::SymInt, c10::SymInt, c10::SymInt, int64_t, double)>(&habana::native_group_norm));
  m.impl("linear_backward", static_cast<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (*)(const at::Tensor &, const at::Tensor &, const at::Tensor &, ::std::array<bool,3>)>(&habana::linear_backward));

}



}  // namespace habana

