// Autogenerated file by gen_op.py. Do not edit directly!
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_validator.h"
#include "hpu_ops/op_logger.h"
#include "common/dump_args.h"
#include "habana_eager/eager_exec.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/override_fns.h"
#include "_deform_conv2d_backward.h"


using habana_helpers::DTypeHelper;
using synapse_helpers::graph;
using torch::jit::Stack;


namespace habana {



::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _deform_conv2d_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & offset, const at::Tensor & mask, const at::Tensor & bias, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dilation_h, int64_t dilation_w, int64_t groups, int64_t offset_groups, bool use_mask) {
  PT_EAGER_TRACE;
  PT_OP_INFO("_deform_conv2d_backward: ", DUMP_15ARGS(grad, input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups, use_mask));

  [[maybe_unused]] bool require_h2d = false;
  [[maybe_unused]] bool require_st = false;

  eager::EagerOp<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>> hpu_op{"torchvision::_deform_conv2d_backward", {grad, input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups, use_mask}};
  hpu_op.SetOutputMetaFn(DeformConv2dBackwardOutputMeta);
  hpu_op.set_eager_op_info({eager::eagerOpKind::OutOfPlace, "torchvision::_deform_conv2d_backward", require_h2d, require_st, decltype(eager::EagerOpMetaData::out_indices_){}});
  return hpu_op.call();
}





static const auto& kr_gen__torchvision = KernelRegistry()
;

TORCH_LIBRARY_IMPL(torchvision, HPU, m) {
  m.impl("_deform_conv2d_backward", static_cast<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (*)(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool)>(&habana::_deform_conv2d_backward));

}



}  // namespace habana

