// Autogenerated file by gen_op.py. Do not edit directly!
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_validator.h"
#include "hpu_ops/op_logger.h"
#include "common/dump_args.h"
#include "habana_eager/eager_exec.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/override_fns.h"
#include "native_dropout.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {



::std::tuple<at::Tensor,at::Tensor> native_dropout(const at::Tensor & input, double p, c10::optional<bool> train) {
  PT_EAGER_TRACE;
  PT_OP_INFO("native_dropout: ", DUMP_3ARGS(input, p, train));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}}}))
  FALLBACK_IF_UNSUPPORTED_DTYPE(input, native_dropout, input, p, train)

  if (auto eePath = NativeDropoutEarlyExitCondition(input, p, train))
    return NativeDropoutEarlyExit(eePath, input, p, train);

  NativeDropoutFE<::std::tuple<at::Tensor,at::Tensor>> hpu_op{"aten::native_dropout", {input, p, train}};
  hpu_op.SetOutputMetaFn(FusedNativeDropoutMeta);
  hpu_op.set_eager_op_info({eager::eagerOpKind::OutOfPlace, "aten::native_dropout", require_h2d, require_st, decltype(eager::EagerOpMetaData::out_indices_){}});
  return hpu_op.call();
}





static const auto& kr_gen_3 = KernelRegistry()
;

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("native_dropout", static_cast<::std::tuple<at::Tensor,at::Tensor> (*)(const at::Tensor &, double, c10::optional<bool>)>(&habana::native_dropout));

}



}  // namespace habana

