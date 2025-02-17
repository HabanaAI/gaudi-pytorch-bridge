// Autogenerated file by gen_op.py. Do not edit directly!
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_validator.h"
#include "hpu_ops/op_logger.h"
#include "common/dump_args.h"
#include "habana_eager/eager_exec.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/override_fns.h"
#include "prod.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {



at::Tensor & prod_out(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  PT_EAGER_TRACE;
  PT_OP_INFO("prod_out: ", DUMP_5ARGS(self, dim, keepdim, dtype, out));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kChar, at::kByte, at::kShort, at::kInt, at::kDouble, at::kBool}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kChar, at::kByte, at::kShort, at::kInt, at::kHalf, at::kDouble, at::kBool}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kChar, at::kByte, at::kShort, at::kInt, at::kHalf, at::kDouble, at::kBool}}}))
  FALLBACK_IF_UNSUPPORTED_DTYPE2(self, prod, int_out, self, dim, keepdim, dtype, out)

  eager::EagerOp<at::Tensor &> hpu_op{"aten::prod", {self, dim, keepdim, dtype, out}};
  hpu_op.set_eager_op_info({eager::eagerOpKind::InplaceOut, "aten::prod", require_h2d, require_st, 1});
  return hpu_op.call(out);
}





static const auto& kr_gen_7 = KernelRegistry()
;

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("prod.int_out", static_cast<at::Tensor & (*)(const at::Tensor &, int64_t, bool, c10::optional<at::ScalarType>, at::Tensor &)>(&habana::prod_out));

}



}  // namespace habana

