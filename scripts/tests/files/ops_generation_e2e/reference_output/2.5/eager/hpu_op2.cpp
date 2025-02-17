// Autogenerated file by gen_op.py. Do not edit directly!
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_validator.h"
#include "hpu_ops/op_logger.h"
#include "common/dump_args.h"
#include "habana_eager/eager_exec.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/override_fns.h"
#include "_fused_dropout.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {



::std::tuple<at::Tensor,at::Tensor> _fused_dropout(const at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  PT_EAGER_TRACE;
  PT_OP_INFO("_fused_dropout: ", DUMP_3ARGS(self, p, generator));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}}}))
  FALLBACK_IF_UNSUPPORTED_DTYPE(self, _fused_dropout, self, p, generator)

  GeneratorToSeed<::std::tuple<at::Tensor,at::Tensor>> hpu_op{"aten::_fused_dropout", {self, p, generator}};
  hpu_op.SetOutputMetaFn(FusedNativeDropoutMeta);
  hpu_op.set_eager_op_info({eager::eagerOpKind::OutOfPlace, "aten::_fused_dropout", require_h2d, require_st, decltype(eager::EagerOpMetaData::out_indices_){}});
  return hpu_op.call();
}





static const auto& kr_gen_2 = KernelRegistry()
;

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("_fused_dropout", static_cast<::std::tuple<at::Tensor,at::Tensor> (*)(const at::Tensor &, double, c10::optional<at::Generator>)>(&habana::_fused_dropout));

}



}  // namespace habana

