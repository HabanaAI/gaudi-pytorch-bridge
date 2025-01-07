// Autogenerated file by gen_op.py. Do not edit directly!
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_validator.h"
#include "hpu_ops/op_logger.h"
#include "common/dump_args.h"
#include "habana_eager/eager_exec.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/override_fns.h"
#include "_native_batch_norm_legit.h"
#include "convolution_backward_overrideable.h"
#include "eq.h"
#include "linear_backward.h"
#include "native_group_norm.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {



at::Tensor & eq_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  PT_EAGER_TRACE;
  PT_OP_INFO("eq_out: ", DUMP_3ARGS(self, other, out));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  auto compute_type = DTypeHelper::get_compute_dtype({self, other}, out, DTypeHelper::DtypePromoteVariant::kPromoteToCommon, false/*safe_cast*/);
  static_cast<void>(compute_type);

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kInt, at::kChar, at::kByte, at::kLong, at::kDouble, at::kBool}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kInt, at::kChar, at::kByte, at::kLong, at::kShort, at::kDouble, at::kBool}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kInt, at::kChar, at::kByte, at::kLong, at::kShort, at::kDouble, at::kBool}}}))
  FALLBACK_IF_UNSUPPORTED_DTYPE2(compute_type, eq, Scalar_out, self, other, out)

  eager::EagerOp<at::Tensor &> hpu_op{"aten::eq", {self, other, out}};
  hpu_op.set_scalar_types({compute_type});
  hpu_op.SetOutputMetaFn(CompareMeta);
  hpu_op.set_eager_op_info({eager::eagerOpKind::InplaceOut, "aten::eq", require_h2d, require_st, 1});
  return hpu_op.call(out);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, bool training, double momentum, double eps) {
  PT_EAGER_TRACE;
  PT_OP_INFO("_native_batch_norm_legit: ", DUMP_8ARGS(input, weight, bias, running_mean, running_var, training, momentum, eps));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}}}), input)
  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kFloat, at::kDouble}},
   {synDeviceGaudi3, {at::kFloat, at::kDouble}}}), weight)
  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kFloat, at::kDouble}},
   {synDeviceGaudi3, {at::kFloat, at::kDouble}}}), bias)
  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kFloat, at::kDouble}},
   {synDeviceGaudi3, {at::kFloat, at::kDouble}}}), running_mean)
  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kFloat, at::kDouble}},
   {synDeviceGaudi3, {at::kFloat, at::kDouble}}}), running_var)
  FALLBACK_IF_UNSUPPORTED_DTYPE_PER_TENSOR(input, _native_batch_norm_legit, input, weight, bias, running_mean, running_var, training, momentum, eps)
  FALLBACK_IF_UNSUPPORTED_DTYPE_PER_TENSOR(running_mean, _native_batch_norm_legit, input, weight, bias, running_mean, running_var, training, momentum, eps)
  FALLBACK_IF_UNSUPPORTED_DTYPE_PER_TENSOR(running_var, _native_batch_norm_legit, input, weight, bias, running_mean, running_var, training, momentum, eps)

  eager::EagerOp<::std::tuple<at::Tensor,at::Tensor,at::Tensor>> hpu_op{"aten::_native_batch_norm_legit", {input, weight, bias, running_mean, running_var, training, momentum, eps}};
  hpu_op.SetOutputMetaFn(BatchNormFwdMeta);
  hpu_op.set_eager_op_info({eager::eagerOpKind::OutOfPlace, "aten::_native_batch_norm_legit", require_h2d, require_st, decltype(eager::EagerOpMetaData::out_indices_){}});
  return hpu_op.call();
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, ::std::array<bool,3> output_mask) {
  PT_EAGER_TRACE;
  PT_OP_INFO("convolution_backward_overrideable: ", DUMP_10ARGS(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}}}))
  FALLBACK_IF_UNSUPPORTED_DTYPE(grad_output, convolution_backward_overrideable, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask)
  FALLBACK_IF_UNSUPPORTED_DTYPE(input, convolution_backward_overrideable, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask)
  FALLBACK_IF_UNSUPPORTED_DTYPE(weight, convolution_backward_overrideable, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask)

  ConvolutionBackwardOverrideableFE<::std::tuple<at::Tensor,at::Tensor,at::Tensor>> hpu_op{"aten::convolution_backward_overrideable", {grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask}};
  hpu_op.SetOutputMetaFn(ConvolutionOverrideableMetaBwd);
  hpu_op.set_eager_op_info({eager::eagerOpKind::OutOfPlace, "aten::convolution_backward_overrideable", require_h2d, require_st, decltype(eager::EagerOpMetaData::out_indices_){}});
  return hpu_op.call();
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, double eps) {
  PT_EAGER_TRACE;
  PT_OP_INFO("native_group_norm: ", DUMP_8ARGS(input, weight, bias, N, C, HxW, group, eps));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kDouble}}}))
  FALLBACK_IF_UNSUPPORTED_DTYPE(input, native_group_norm, input, weight, bias, N, C, HxW, group, eps)

  eager::EagerOp<::std::tuple<at::Tensor,at::Tensor,at::Tensor>> hpu_op{"aten::native_group_norm", {input, weight, bias, N, C, HxW, group, eps}};
  hpu_op.SetOutputMetaFn(GroupNormFwdMeta);
  hpu_op.set_eager_op_info({eager::eagerOpKind::OutOfPlace, "aten::native_group_norm", require_h2d, require_st, decltype(eager::EagerOpMetaData::out_indices_){}});
  return hpu_op.call();
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> linear_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, ::std::array<bool,3> output_mask) {
  PT_EAGER_TRACE;
  PT_OP_INFO("linear_backward: ", DUMP_4ARGS(self, grad_output, weight, output_mask));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}}}))
  FALLBACK_IF_UNSUPPORTED_DTYPE(self, linear_backward, self, grad_output, weight, output_mask)
  FALLBACK_IF_UNSUPPORTED_DTYPE(grad_output, linear_backward, self, grad_output, weight, output_mask)
  FALLBACK_IF_UNSUPPORTED_DTYPE(weight, linear_backward, self, grad_output, weight, output_mask)

  eager::EagerOp<::std::tuple<at::Tensor,at::Tensor,at::Tensor>> hpu_op{"aten::linear_backward", {self, grad_output, weight, output_mask}};
  hpu_op.SetOutputMetaFn(LinearBackwardMeta);
  hpu_op.set_eager_op_info({eager::eagerOpKind::OutOfPlace, "aten::linear_backward", require_h2d, require_st, decltype(eager::EagerOpMetaData::out_indices_){}});
  return hpu_op.call();
}





static const auto& kr_gen_9 = KernelRegistry()
;

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("eq.Scalar_out", static_cast<at::Tensor & (*)(const at::Tensor &, const at::Scalar &, at::Tensor &)>(&habana::eq_out));
  m.impl("_native_batch_norm_legit", static_cast<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (*)(const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, at::Tensor &, at::Tensor &, bool, double, double)>(&habana::_native_batch_norm_legit));
  m.impl("convolution_backward_overrideable", static_cast<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (*)(const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::SymIntArrayRef, c10::SymIntArrayRef, c10::SymIntArrayRef, bool, c10::SymIntArrayRef, c10::SymInt, ::std::array<bool,3>)>(&habana::convolution_backward_overrideable));
  m.impl("native_group_norm", static_cast<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (*)(const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, c10::SymInt, c10::SymInt, c10::SymInt, int64_t, double)>(&habana::native_group_norm));
  m.impl("linear_backward", static_cast<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (*)(const at::Tensor &, const at::Tensor &, const at::Tensor &, ::std::array<bool,3>)>(&habana::linear_backward));

}



}  // namespace habana

