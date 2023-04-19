
/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "backend/backend_meta.h"
#include "habana_eager/helpers.h"
#include "habana_eager/ops/as_strided.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/empty.h"
#include "habana_eager/ops/set.h"
#include "habana_eager/ops/view.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/op_logger.h"

#include "habana_helpers/logging.h"

using namespace at;
using namespace habana;

Tensor hpu_wrap::empty(
    SymIntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  PT_EAGER_TRACE;
  return habana::eager::empty(
      size, dtype, layout, device, pin_memory, optional_memory_format);
}

Tensor hpu_wrap::empty_strided(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  PT_EAGER_TRACE;
  return habana::eager::empty_strided(
      size, stride, dtype, layout, device, pin_memory);
}

at::Tensor depermute_dims(
    const at::Tensor t,
    const std::vector<unsigned char>& synapse_permute) {
  size_t dim_count{synapse_permute.size()};
  std::vector<int64_t> permute_dims(dim_count, -1);

  // start by conversion to torch layout
  //   first reverse vector and then change idx to mirror
  //   NHWC 2013 -> 3102 -> 0231
  // then generate sequence that will reverse the permutation
  //   where(0)=0, where(1)=3, where(2)=1, where(3)=2 -> 0312
  for (size_t no = 0; no < synapse_permute.size(); ++no)
    permute_dims[dim_count - 1 - synapse_permute[no]] = dim_count - no - 1;

  habana::eager::EagerOp<at::Tensor> hpu_op{"aten::permute", {t, permute_dims}};
  return hpu_op.call();
}

Tensor hpu_wrap::_reshape_alias(
    const Tensor& self,
    SymIntArrayRef size,
    SymIntArrayRef stride) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "_reshape_alias :",
      " self=",
      to_string(self),
      " size=",
      to_string(size),
      " stride",
      to_string(stride));
  FALLBACK_IF_UNSUPPORTED_OP(
      _reshape_alias, PARAMS1(self), PARAMS2(self, size, stride))
  auto tmeta{habana::get_tensor_extra_meta(self)};
  const auto& synapse_permute{tmeta->get_memory_permutation()};
  if (synapse_permute.empty())
    return habana::eager::view_hpu(self, size);
  return habana::eager::view_hpu(depermute_dims(self, synapse_permute), size);
}

at::Tensor fused_norm_hpu_wrap(
    std::vector<at::Tensor>& grad,
    const at::Tensor& max_norm,
    float norm_type) {
  auto FusedNormMeta = [](const at::Stack& stack) {
    OutputMetaDataVector meta_vec;
    meta_vec.resize(stack[0].toTensorList().size() + 1);
    return meta_vec;
  };

  habana::eager::EagerOp<std::vector<at::Tensor>> hpu_op{
      "hpu::fused_norm_lazy", {grad, max_norm, norm_type}};
  hpu_op.SetOutputMeta(FusedNormMeta);
  auto res = hpu_op.call();
  for (int i = 0; i < grad.size(); ++i) {
    grad[i].copy_(res[i + 1]);
  }
  return res[0];
}
