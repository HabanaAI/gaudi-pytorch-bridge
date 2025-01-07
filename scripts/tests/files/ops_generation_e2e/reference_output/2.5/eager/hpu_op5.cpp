// Autogenerated file by gen_op.py. Do not edit directly!
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_validator.h"
#include "hpu_ops/op_logger.h"
#include "common/dump_args.h"
#include "habana_eager/eager_exec.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/override_fns.h"
#include "addbmm.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {

static CheckNodeWithSharedLayerValidator validator_addbmm("addbmm", AddBMMSharedMeta, habana_helpers::HabanaExecutionMode::EAGER);


at::Tensor addbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  PT_EAGER_TRACE;
  PT_OP_INFO("addbmm: ", DUMP_5ARGS(self, batch1, batch2, beta, alpha));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  VAL_CUSTOM_FALLBACK_IF_UNSUPPORTED_DTYPE(addbmm, true, self, batch1, batch2, beta, alpha)

  eager::EagerOp<at::Tensor> hpu_op{"aten::addbmm", {self, batch1, batch2, beta, alpha}};
  hpu_op.SetOutputMetaFn(AddBMMMeta);
  hpu_op.set_eager_op_info({eager::eagerOpKind::OutOfPlace, "aten::addbmm", require_h2d, require_st, decltype(eager::EagerOpMetaData::out_indices_){}});
  return hpu_op.call();
}





static const auto& kr_gen_5 = KernelRegistry()
;

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("addbmm", static_cast<at::Tensor (*)(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &)>(&habana::addbmm));

}



}  // namespace habana

