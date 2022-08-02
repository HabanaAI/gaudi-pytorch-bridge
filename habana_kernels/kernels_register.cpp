/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/library.h>

#include <torch/csrc/api/include/torch/version.h>
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/lazy_executor.h"
#include "hpu_ops/cpu_fallback.h"
#include "kernel_input_checks.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"
#include "pytorch_helpers/pt_ver/torch_params_shim.h"
#include "synapse_helpers/env_flags.h"

using namespace torch;
using namespace at;
using namespace habana;
using namespace habana_lazy;

Tensor hpu_wrap::_to_copy(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<MemoryFormat> optional_memory_format) {
  PT_OP_INFO(
      "_to_copy :",
      " self=",
      to_string(self),
      " dtype=",
      to_string(dtype),
      " layout=",
      to_string(layout),
      " device=",
      to_string(device),
      " pin_memory=",
      to_string(pin_memory),
      " non_blocking=",
      to_string(non_blocking),
      " optional_memory_format=",
      to_string(optional_memory_format));
  habana_lazy::SyncAccThreadPool();
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);
  auto options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  options = self.options().merge_in(options);
  if (memory_format == MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      // Copy all strides
      auto r = at::empty_strided(
          self.sizes(), self.strides(), options.memory_format(c10::nullopt));
      r.copy_(self, non_blocking);
      return r;
    } else {
      memory_format = self.suggest_memory_format();
    }
  }

  auto r = at::empty(
      self.sizes(), options.memory_format(memory_format), c10::nullopt);
  r.copy_(self, non_blocking);
  return r;
}

bool hpu_wrap::is_pinned(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  PT_OP_INFO(
      "is_pinned :", " self=", to_string(self), " device=", to_string(device));
  return is_pinned_hpu(self, device);
}

Tensor hpu_wrap::pin_memory(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  PT_OP_INFO(
      "pin_memory :", " self=", to_string(self), " device=", to_string(device));
  return pin_memory_hpu(self, device);
};

Tensor hpu_wrap::_pin_memory(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  PT_OP_INFO(
      "_pin_memory :",
      " self=",
      to_string(self),
      " device=",
      to_string(device));
  return pin_memory_hpu(self, device);
};
/*
 PT1.12 introduced a change in linear() to use addmm instead of matmul
 in case of 3d input. This caused a perf regression on HPU. In order to
 circumvent this regression we register linear() ( a compound op) so that
 it gets dispatched as such to HPU. We use essentially the same linear()
 impl. as in PyTorch but removing the PT1.12 change and other code not
 relevant to HPU. Ref. aten/src/ATen/native/Linear.cpp
 Ref. https://jira.habana-labs.com/browse/SW-93519 for details.
*/
Tensor linear_(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  PT_OP_INFO(
      "HpuOp linear:",
      " input=",
      to_string(input),
      " weight=",
      to_string(weight),
      " bias_opt=",
      to_string(bias_opt));
  auto bias = bias_opt.has_value()
      ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
      : c10::MaybeOwned<Tensor>::owned(c10::in_place);

  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm(*bias, input, weight.t());
  }
  return linear_non2d_hpu_lazy(input, weight, bias_opt);
}

Tensor& hpu_wrap::copy_(Tensor& self, const Tensor& src_, bool non_blocking) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "copy_ :",
      " self=",
      to_string(self),
      " src=",
      to_string(src_),
      " non_blocking=",
      to_string(non_blocking));
  Tensor src = src_;
  if (src.device().type() == c10::DeviceType::HPU &&
      self.device().type() == c10::DeviceType::HPU) {
    if (src.scalar_type() == c10::ScalarType::Float &&
        self.scalar_type() == c10::ScalarType::Byte) {
      FALLBACK_IF_UNSUPPORTED_OP2(copy_, PARAMS2(self, src, non_blocking))
    }
    // WA: There is no direct bf16 to bool(i8) cast available in gaudi.
    if (src.scalar_type() == c10::ScalarType::BFloat16 &&
        self.scalar_type() == c10::ScalarType::Bool) {
      src = src_.to(c10::ScalarType::Float);
    }
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    habana_lazy::SyncAccThreadPool();
    return copy_hpu_lazy_(self, src, non_blocking);
  } else {
    if (src.device().type() == c10::DeviceType::HPU &&
        self.device().type() == c10::DeviceType::HPU) {
      if (src.scalar_type() == c10::ScalarType::Float &&
          self.scalar_type() == c10::ScalarType::Long) {
        FALLBACK_IF_UNSUPPORTED_OP2(copy_, PARAMS2(self, src, non_blocking))
      }
    }
    return copy_hpu_(self, src, non_blocking);
  }
};

Tensor hpu_wrap::_reshape_alias(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride) {
  PT_OP_TRACE;
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
  // TODO: In order to align the changes of bert with Pytorchv1.9 we used
  // view inplace of as_strided implementation for the reshape of tensor
  // with no-change.
  // We need to revert existing change and use only as_strided once we
  // establish the convergence with below changes.
  // Pytorch change: https://github.com/pytorch/pytorch/pull/61466
  // Below is the proposed change:
  // if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
  //  return as_strided_hpu_lazy(self, size, stride, c10::nullopt);
  //
  //} else {
  //  return as_strided_hpu(self, size, stride, c10::nullopt);
  //}
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return view_hpu_lazy(self, size);

  } else {
    return view_hpu(self, size);
  }
};

Tensor hpu_wrap::as_strided(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  // No CPU fallback for as_strided since H2D & D2H DMA support only contiguous
  // tensor transfers
  // FALLBACK_IF_UNSUPPORTED_OP(__func__, PARAMS1(self), PARAMS2(self, size,
  // stride, storage_offset))
  PT_OP_TRACE;
  PT_OP_INFO(
      "as_strided :",
      " self=",
      to_string(self),
      " size=",
      to_string(size),
      " stride=",
      to_string(stride),
      " storage_offset=",
      to_string(storage_offset));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return as_strided_hpu_lazy(self, size, stride, storage_offset);

  } else {
    return as_strided_hpu(self, size, stride, storage_offset);
  }
};
Tensor& hpu_wrap::set_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  PT_OP_TRACE;
  FALLBACK_IF_UNSUPPORTED_OP_O(
      set_,
      PARAMS1(self),
      PARAMS2(self, source, storage_offset, size, stride),
      source_Storage_storage_offset)
  return set_hpu_lazy_(self, source, storage_offset, size, stride);
}

Tensor hpu_wrap::all(const Tensor& self, int64_t dim, bool keepdim) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "all :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " keepdim=",
      to_string(keepdim));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      all, PARAMS1(self), PARAMS2(self, dim, keepdim), dim)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return all_dim_hpu_lazy(self, dim, keepdim);
  } else {
    return all_dim_hpu(self, dim, keepdim);
  }
};

Tensor hpu_wrap::constant_pad_nd(
    const Tensor& self,
    IntArrayRef pad,
    const Scalar& value) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "constant_pad :",
      " self=",
      to_string(self),
      " pad=",
      to_string(pad),
      " value=",
      to_string(value));
  FALLBACK_IF_UNSUPPORTED_OP(
      constant_pad_nd, PARAMS1(self), PARAMS2(self, pad, value))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return constant_pad_hpu_lazy(self, pad, value);

  } else {
    return constant_pad_hpu(self, pad, value);
  }
};
Tensor embedding_bag_sum_hpu_wrap(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "embedding_bag_sum :",
      " input=",
      to_string(input),
      " indices=",
      to_string(indices),
      " offsets=",
      to_string(offsets),
      " valid_count=",
      to_string(valid_count),
      " kernel_mode=",
      to_string(kernel_mode));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return embedding_bag_sum_hpu_lazy(
        input, indices, offsets, valid_count, kernel_mode);

  } else {
    return embedding_bag_sum_hpu(
        input, indices, offsets, valid_count, kernel_mode);
  }
};
Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "embedding_bag_sum_bwd_out :",
      " out=",
      to_string(out),
      " input=",
      to_string(input),
      " indices=",
      to_string(indices),
      " offsets=",
      to_string(offsets),
      " valid_count=",
      to_string(valid_count),
      " kernel_mode=",
      to_string(kernel_mode));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return embedding_bag_sum_bwd_out_kernel_mode_hpu_lazy(
        out, input, indices, offsets, valid_count, kernel_mode);

  } else {
    return embedding_bag_sum_bwd_out_kernel_mode_hpu(
        out, input, indices, offsets, valid_count, kernel_mode);
  }
};
Tensor& hpu_wrap::masked_fill_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "masked_fill_ :",
      " self=",
      to_string(self),
      " mask=",
      to_string(mask),
      " value=",
      to_string(value));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      masked_fill_,
      PARAMS1(self, mask, value),
      PARAMS2(self, mask, value),
      Tensor)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return masked_fill_hpu_lazy_(self, mask, value);

  } else {
    return masked_fill_hpu_(self, mask, value);
  }
};
Tensor& hpu_wrap::masked_fill_(
    Tensor& self,
    const Tensor& mask,
    const Scalar& value) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "masked_fill_ :",
      " self=",
      to_string(self),
      " mask=",
      to_string(mask),
      " value=",
      to_string(value));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      masked_fill_, PARAMS1(self, mask), PARAMS2(self, mask, value), Scalar)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return masked_fill_scalar_hpu_lazy_(self, mask, value);

  } else {
    return masked_fill_scalar_hpu_(self, mask, value);
  }
};
Tensor hpu_wrap::masked_select(const Tensor& self, const Tensor& mask) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "masked_select :", " self=", to_string(self), " mask=", to_string(mask));
  FALLBACK_IF_UNSUPPORTED_OP(
      masked_select, PARAMS1(self, mask), PARAMS2(self, mask))
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return masked_select_hpu_lazy(self, mask);
  } else {
    HABANA_ASSERT(0 && "masked_select not implemented for eager mode");
    return masked_select_hpu_lazy(self, mask);
  }
};
Tensor& hpu_wrap::masked_select_out(
    const Tensor& self,
    const Tensor& mask,
    Tensor& out) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "masked_select_out :",
      " self=",
      to_string(self),
      " mask=",
      to_string(mask),
      " out=",
      to_string(out));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      masked_select, PARAMS1(self, mask, out), PARAMS2(self, mask, out), out)
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return masked_select_out_hpu_lazy(self, mask, out);
  } else {
    HABANA_ASSERT(0 && "masked_select_out not implemented for eager mode");
    return masked_select_out_hpu_lazy(self, mask, out);
  }
};

Tensor hpu_wrap::scatter_add(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "scatter_add :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim_),
      " index=",
      to_string(index),
      " src=",
      to_string(src));
  FALLBACK_IF_UNSUPPORTED_OP(
      scatter_add, PARAMS1(self, index, src), PARAMS2(self, dim_, index, src))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return scatter_add_src_hpu_lazy(self, dim_, index, src);

  } else {
    return scatter_add_src_hpu(self, dim_, index, src);
  }
};
Tensor& hpu_wrap::scatter_add_(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "scatter_add_ :",
      " self=",
      to_string(self),
      " dim_=",
      to_string(dim_),
      " index=",
      to_string(index),
      " src=",
      to_string(src));
  FALLBACK_IF_UNSUPPORTED_OP(
      scatter_add_, PARAMS1(self, index, src), PARAMS2(self, dim_, index, src))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return scatter_add_inplace_src_hpu_lazy(self, dim_, index, src);

  } else {
    return scatter_add_inplace_src_hpu(self, dim_, index, src);
  }
};

Tensor& hpu_wrap::index_add_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    Tensor& out) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "index_add_out :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " index=",
      to_string(index),
      " source=",
      to_string(source),
      " alpha=",
      to_string(alpha),
      " out=",
      to_string(out));
  FALLBACK_IF_UNSUPPORTED_OP(
      index_add_out,
      PARAMS1(self, index, source, out),
      PARAMS2(self, dim, index, source, alpha, out))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return index_add_hpu_lazy_out(self, dim, index, source, alpha, out);
  } else {
    HABANA_ASSERT(0 && "index_add_out is not implemented for eager mode");
    /* dummy return to satisfy compiler */
    return out;
  }
}

Tensor& hpu_wrap::masked_scatter_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& source) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "masked_scatter_ :",
      " self=",
      to_string(self),
      " mask=",
      to_string(mask),
      " source=",
      to_string(source));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return masked_scatter_hpu_lazy_(self, mask, source);
  } else {
    HABANA_ASSERT(0 && "masked_scatter is not implemented for eager mode");
    return masked_scatter_hpu_lazy_(self, mask, source);
  }
};

Tensor& hpu_wrap::index_fill_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "index_fill_ :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      "index=",
      to_string(index),
      "value=",
      to_string(value));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return index_fill_hpu_lazy_(self, dim, index, value);
  } else {
    HABANA_ASSERT(0 && "index_fill_ is not implemented for eager mode");
    return index_fill_hpu_lazy_(self, dim, index, value);
  }
};

Tensor& hpu_wrap::index_copy_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& value) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "index_copy_ :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      "index=",
      to_string(index),
      "value=",
      to_string(value));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return index_copy_hpu_lazy_(self, dim, index, value);
  } else {
    HABANA_ASSERT(0 && "index_copy_ is not implemented for eager mode");
    return index_copy_hpu_lazy_(self, dim, index, value);
  }
};

Tensor hpu_wrap::select_backward(
    const at::Tensor& grad,
    at::IntArrayRef input_sizes,
    int64_t dim,
    int64_t index) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "select_backward :",
      " grad=",
      to_string(grad),
      " input_sizes=",
      to_string(input_sizes),
      " dim=",
      to_string(dim),
      " index=",
      to_string(index));
  FALLBACK_IF_UNSUPPORTED_OP(
      select_backward, PARAMS1(grad), PARAMS2(grad, input_sizes, dim, index))

  return select_backward_hpu_lazy(grad, input_sizes, dim, index);
};
Tensor& hpu_wrap::arange_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& output) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "arange_out:",
      " start=",
      to_string(start),
      " end=",
      to_string(end),
      " step=",
      to_string(step),
      " output=",
      to_string(output));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      arange, PARAMS1(output), PARAMS2(start, end, step, output), start_out)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return arange_hpu_lazy(output, start, end, step);

  } else {
    return arange_hpu(output, start, end, step);
  }
};
Tensor& hpu_wrap::nonzero_out(const Tensor& self, Tensor& out) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "nonzero_out :", " self=", to_string(self), " out=", to_string(out));
  FALLBACK_IF_UNSUPPORTED_OP(
      nonzero_out, PARAMS1(self, out), PARAMS2(self, out))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return nonzero_out_hpu_lazy(self, out);
  } else {
    HABANA_ASSERT(0 && "nonzero_out is not implemented for eager mode");
    return nonzero_out_hpu_lazy(self, out);
  }
};

Tensor hpu_wrap::baddbmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "baddbmm :",
      " self=",
      to_string(self),
      " mat1=",
      to_string(mat1),
      " mat2=",
      to_string(mat2),
      " beta=",
      to_string(beta),
      " alpha=",
      to_string(alpha));
  Tensor out = torch::mul(torch::bmm(mat1, mat2), alpha);
  if (beta.toFloat() != 0) {
    out.add_(self, beta);
  }
  return out;
}

Tensor& hpu_wrap::baddbmm_out(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "baddbmm_out :",
      " self=",
      to_string(self),
      " mat1=",
      to_string(mat1),
      " mat2=",
      to_string(mat2),
      " beta=",
      to_string(beta),
      " alpha=",
      to_string(alpha),
      " out=",
      to_string(out));
  if (beta.toFloat() == 0) {
    torch::bmm_outf(mat1, mat2, out);
    out.mul_(alpha);
  } else {
    Tensor r_bmul = torch::mul(self, beta);
    torch::bmm_outf(mat1, mat2, out);
    out.mul_(alpha);
    out.add_(r_bmul, 1);
  }
  return out;
}

Tensor& hpu_wrap::baddbmm_(
    Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "baddbmm_ :",
      " self=",
      to_string(self),
      " mat1=",
      to_string(mat1),
      " mat2=",
      to_string(mat2),
      " beta=",
      to_string(beta),
      " alpha=",
      to_string(alpha));
  if (beta.toFloat() == 0) {
    torch::bmm_outf(mat1, mat2, self);
    self.mul_(alpha);
  } else {
    Tensor r_bmm = torch::bmm(mat1, mat2);
    self.mul_(beta);
    self.add_(r_bmm, alpha);
  }
  return self;
}

Tensor hpu_wrap::kl_div_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "kl_div_backward :",
      " grad=",
      to_string(grad),
      " self=",
      to_string(self),
      " target=",
      to_string(target),
      " reduction=",
      to_string(reduction),
      " log_target=",
      to_string(log_target));
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(grad),
      IValue(self),
      IValue(target),
      IValue(reduction),
      IValue(log_target)};
  check_handle->hpu_check_ivalues("kl_div_backward", op_stack);

#if ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR < 13))
  FALLBACK_IF_UNSUPPORTED_OP1(
      kl_div_backward,
      PARAMS1(grad, self, target),
      PARAMS2(grad, self, target, reduction, log_target))
#endif

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return kl_div_backward_hpu_lazy(grad, self, target, reduction, log_target);
  } else {
    return kl_div_backward_hpu(grad, self, target, reduction, log_target);
  }
};

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::batch_norm_stats(
    const at::Tensor& input,
    double eps) {
  return batch_norm_stats_lazy(input, eps);
}
at::Tensor hpu_wrap::batch_norm_elemt(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps) {
  return batch_norm_elemt_lazy(input, weight, bias, mean, invstd, eps);
}
at::Tensor hpu_wrap::batch_norm_backward_elemt(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& weight,
    const at::Tensor& mean_dy,
    const at::Tensor& mean_dy_xmu,
    const at::Tensor& count) {
  return batch_norm_backward_elemt_lazy(
      grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hpu_wrap::
    batch_norm_backward_reduce(
        const at::Tensor& grad_out,
        const at::Tensor& input,
        const at::Tensor& mean,
        const at::Tensor& invstd,
        const c10::optional<at::Tensor>& weight,
        bool input_g,
        bool weight_g,
        bool bias_g) {
  return batch_norm_backward_reduce_lazy(
      grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::
    batch_norm_gather_stats_with_counts(
        const at::Tensor& input,
        const at::Tensor& mean,
        const at::Tensor& invstd,
        const c10::optional<at::Tensor>& running_mean,
        const c10::optional<at::Tensor>& running_var,
        double momentum,
        double eps,
        const at::Tensor& counts) {
  return batch_norm_gather_stats_with_counts_lazy(
      input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

std::tuple<Tensor, Tensor> hpu_wrap::_weight_norm_interface(
    const Tensor& v_in,
    const Tensor& g_in,
    int64_t dim) {
  /*
  NOTE:
  We use the CPU implementation that follows the "non-fused" (ie., assumes
  can_use_fused=0) path.
  */
  TORCH_CHECK(
      v_in.device() == g_in.device(),
      "weight_norm: expected v_in and g_in to be on the same device, but v_in is "
      "on ",
      v_in.device(),
      " and g_in is on ",
      g_in.device());
  auto v = v_in.contiguous();
  auto g = g_in.contiguous();
  // align with cuda behavior, keep norm in 'Float' when g is 'BFloat16'
  const auto dtype = (g.scalar_type() == at::ScalarType::BFloat16)
      ? at::ScalarType::Float
      : g.scalar_type();
  auto norm = at::norm_except_dim(v.to(dtype), 2, dim);
  // Double-differentiable primitive ops
  // at::native::norm_except_dim would probably be fine as well.
  return std::make_tuple(v * (g / norm), norm);
}

std::tuple<Tensor, Tensor> hpu_wrap::_weight_norm_interface_backward(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norms,
    int64_t dim) {
  /*
  NOTE: Implementation taken as such from
  pytorch/aten/src/ATen/native/WeightNorm.cpp
  */
  // In Functions.cpp, the HardshrinkBackward object supplies
  // "grad.contiguous()" as the first argument, so grad_w should be contiguous
  // here. All these checks should succeed:
  TORCH_CHECK(grad_w.is_contiguous(), "grad_w must be contiguous");
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");

  int64_t last_dim = saved_v.dim() - 1;
  int64_t last_size = saved_v.size(last_dim);

  // Like weight_norm_fused_backward, weight_norm_differentiable_backward should
  // only ever be called through a WeightNormFusedBackward object, so we expect
  // that dim == 0 || dim == saved_v.size(-1)
  TORCH_CHECK(
      dim == 0 || dim == last_dim,
      "Expected dim to be the first or last dimension");

  // saved_g and saved_norms are already shaped to broadcast over the correct
  // dimensions

  // ...but saved_norms might be Float when saved_g and saved_v are half.
  // To consider:  saved_norms.to(..., True /*non_blocking*/);
  auto norms = saved_norms.to(saved_g.scalar_type());

  std::vector<int64_t> bcast_size(saved_v.dim(), 1);

  // Analytic backward path using differentiable primitive ops
  if (dim == 0) {
    bcast_size[0] = saved_v.size(0);
    auto per_dim_sums =
        (grad_w * saved_v).view({saved_v.size(0), -1}).sum(1).view(bcast_size);
    auto grad_v = (saved_g / norms) *
        (grad_w - saved_v * (per_dim_sums / (norms * norms)));
    auto grad_g = per_dim_sums / norms;
    return std::make_tuple(grad_v, grad_g);
  } else { // dim == last_dim
    bcast_size[last_dim] = last_size;
    auto per_dim_sums =
        (grad_w * saved_v).view({-1, last_size}).sum(0).view(bcast_size);
    auto grad_v = (saved_g / norms) *
        (grad_w - saved_v * (per_dim_sums / (norms * norms)));
    auto grad_g = per_dim_sums / norms;
    return std::make_tuple(grad_v, grad_g);
  }
}

Tensor hpu_wrap::instance_norm(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    UNUSED bool use_input_stats,
    UNUSED double momentum,
    double eps,
    UNUSED bool cudnn_enabled) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "instance_norm :",
      " input=",
      to_string(input),
      " weight_opt=",
      to_string(weight_opt),
      " bias_opt=",
      to_string(bias_opt),
      " running_mean_opt=",
      to_string(running_mean_opt),
      " running_var_opt=",
      to_string(running_var_opt),
      " use_input_stats=",
      to_string(use_input_stats),
      " momentum=",
      to_string(momentum),
      " eps=",
      to_string(eps),
      " cudnn_enabled=",
      to_string(cudnn_enabled));
  // Note: Legacy eager mode is not supported
  auto weight = weight_opt.value_or(Tensor());
  auto bias = bias_opt.value_or(Tensor());

  TORCH_CHECK(weight.defined(), "undefined weight is not supported");
  TORCH_CHECK(bias.defined(), "undefined bias is not supported");

  auto running_mean = running_mean_opt.value_or(Tensor());
  auto running_var = running_var_opt.value_or(Tensor());

  if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_INSTANCE_NORM)) {
    return at::native::instance_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        use_input_stats,
        momentum,
        eps,
        cudnn_enabled);
  }

  struct InstanceNorm : public torch::autograd::Function<InstanceNorm> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const Tensor& input,
        const Tensor& weight, // gamma
        const Tensor& bias, // beta
        double eps) {
      Tensor output, mean, istd;
      std::tie(output, mean, istd) =
          instance_norm_hpu_lazy(input, weight, bias, eps);

      ctx->save_for_backward({input, mean, istd, weight});

      return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_in) {
      auto saved = ctx->get_saved_variables();
      auto input = saved[0];
      auto mean = saved[1];
      auto istd = saved[2];
      auto gamma = saved[3];

      Tensor grad_out, grad_beta, grad_gamma;

      std::tie(grad_out, grad_beta, grad_gamma) =
          instance_norm_backward_hpu_lazy(input, grad_in[0], mean, istd, gamma);

      // Autograds same number of gradients as the number of forward inputs and
      // in the same order
      //  grad_eps
      auto grad_eps = Tensor();
      return {grad_out, grad_gamma, grad_beta, grad_eps};
    }
  };

  return InstanceNorm::apply(input, weight, bias, eps);
}

Tensor& hpu_wrap::max_pool2d_with_indices_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices,
    Tensor& grad_input) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "max_pool2d_with_indices_backward_out :",
      " grad_output=",
      to_string(grad_output),
      " input=",
      to_string(input),
      " kernel_size=",
      to_string(kernel_size),
      " stride=",
      to_string(stride),
      " padding=",
      to_string(padding),
      " dilation=",
      to_string(dilation),
      " ceil_mode=",
      to_string(ceil_mode),
      " indices=",
      to_string(indices),
      " grad_input=",
      to_string(grad_input));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      max_pool2d_with_indices_backward,
      PARAMS1(grad_input, grad_output, input, indices),
      PARAMS2(
          grad_output,
          input,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          indices,
          grad_input),
      grad_input)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return max_pool2d_with_indices_backward_out_hpu_lazy(
        grad_input,
        grad_output,
        input,
        indices,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode);
  } else {
    return max_pool2d_with_indices_backward_out_hpu(
        grad_input,
        grad_output,
        input,
        indices,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode);
  }
}

at::Tensor hpu_wrap::repeat(const at::Tensor& self, at::IntArrayRef repeats) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "bernoulli_:", "self=", to_string(self), " repeats=", to_string(repeats));
  FALLBACK_IF_UNSUPPORTED_OP(repeat, PARAMS1(self), PARAMS2(self, repeats))
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return repeat_hpu_lazy(self, repeats);
  } else {
    return repeat_hpu(self, repeats);
  }
}

at::Tensor hpu_wrap::repeat_interleave(
    const at::Tensor& repeats,
    c10::optional<int64_t> output_size) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "repeat_interleave:",
      "repeats=",
      to_string(repeats),
      " output_size=",
      to_string(output_size));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      repeat_interleave,
      PARAMS1(repeats),
      PARAMS2(repeats, output_size),
      Tensor)
  return repeat_inlv_hpu_lazy(repeats, output_size);
}

Tensor hpu_wrap::prod(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "prod :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " keepdim=",
      to_string(keepdim),
      " dtype=",
      to_string(dtype));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      prod, PARAMS1(self), PARAMS2(self, dim, keepdim, dtype), dim_int)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return prod_dim_hpu_lazy(self, dim, keepdim, dtype);

  } else {
    return prod_dim_hpu(self, dim, keepdim, dtype);
  }
};
std::tuple<at::Tensor, at::Tensor> hpu_wrap::max(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "max :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " keepdim=",
      to_string(keepdim));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      max, PARAMS1(self), PARAMS2(self, dim, keepdim), dim)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return max_dim_hpu_lazy(self, dim, keepdim);
  } else {
    return max_dim_hpu(self, dim, keepdim);
  }
};
at::Tensor hpu_wrap::max(const at::Tensor& self) {
  PT_OP_TRACE;
  PT_OP_INFO("max :", " self=", to_string(self));
  FALLBACK_IF_UNSUPPORTED_OP(max, PARAMS1(self), PARAMS2(self))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return max_hpu_lazy(self);
  } else {
    return max_hpu(self);
  }
};

at::Tensor hpu_wrap::min(const at::Tensor& self) {
  PT_OP_TRACE;
  FALLBACK_IF_UNSUPPORTED_OP(min, PARAMS1(self), PARAMS2(self))
  PT_OP_INFO("min :", " self=", to_string(self));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return min_hpu_lazy(self);
  } else {
    return min_hpu(self);
  }
};

Tensor hpu_wrap::one_hot(const Tensor& self, int64_t num_classes) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "one_hot :",
      " self=",
      to_string(self),
      " num_classes=",
      to_string(num_classes));
  FALLBACK_IF_UNSUPPORTED_OP(one_hot, PARAMS1(self), PARAMS2(self, num_classes))
  struct OneHot : public torch::autograd::Function<OneHot> {
    static at::Tensor forward(
        torch::autograd::AutogradContext*,
        const at::Tensor& self,
        int64_t num_classes) {
      return one_hot_hpu_lazy(self, num_classes);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext*,
        const torch::autograd::variable_list&) {
      return {};
    }
  };

  return OneHot::apply(self, num_classes);
}

struct SoftmaxFunction : public torch::autograd::Function<SoftmaxFunction> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor input,
      int64_t dim,
      c10::optional<at::ScalarType> dtype) {
    Tensor converted = dtype.has_value() ? input.toType(dtype.value()) : input;
    auto result = torch::_softmax(converted, dim, false);
    ctx->save_for_backward({result, input});
    ctx->saved_data["dim"] = dim;
    return result;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    auto output = saved_vars[0];
    auto input = saved_vars[1];
    auto dim = ctx->saved_data["dim"].toInt();
    auto result = torch::_softmax_backward_data(
        grad_output[0], output, dim, input.scalar_type());
    return {result, torch::Tensor(), torch::Tensor()};
  }
};

Tensor hpu_wrap::softmax(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "softmax :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " dtype=",
      to_string(dtype));
  return SoftmaxFunction::apply(self, dim, dtype);
}

Tensor hpu_wrap::empty(
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "empty :",
      " size=",
      to_string(size),
      " dtype=",
      to_string(dtype),
      " layout=",
      to_string(layout),
      " device=",
      to_string(device),
      " pin_memory=",
      to_string(pin_memory),
      " optional_memory_format=",
      to_string(optional_memory_format));
  at::TensorOptions options = at::TensorOptions()
                                  .dtype(dtype)
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .device(device);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return empty_hpu_lazy(size, options, optional_memory_format);
  }
  return empty_hpu(size, options, optional_memory_format);
};

Tensor hpu_wrap::empty_strided(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "empty_strided :",
      " size=",
      to_string(size),
      " stride=",
      to_string(stride),
      " dtype=",
      to_string(dtype),
      " layout=",
      to_string(layout),
      " device=",
      to_string(device),
      " pin_memory=",
      to_string(pin_memory));
  FALLBACK_IF_UNSUPPORTED_OP_RT(
      at::dtype_or_default(dtype),
      empty_strided,
      PARAMS1(),
      PARAMS2(
          INTARRAY_PARAM(size),
          INTARRAY_PARAM(stride),
          dtype,
          layout,
          device,
          pin_memory))

  at::TensorOptions options = at::TensorOptions()
                                  .dtype(std::move(dtype))
                                  .layout(std::move(layout))
                                  .pinned_memory(std::move(pin_memory))
                                  .device(std::move(device));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return empty_strided_hpu_lazy(size, stride, options);
  }
  return empty_strided_hpu(size, stride, options);
}

Tensor hpu_wrap::cat(const TensorList tensors, int64_t dim_) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "cat :", " tensors=", to_string(tensors), " dim_=", to_string(dim_));

  FALLBACK_IF_UNSUPPORTED_OP(cat, PARAMS1(tensors[0]), PARAMS2(tensors, dim_))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return cat_hpu_lazy(tensors, dim_);

  } else {
    return cat_hpu(tensors, dim_);
  }
};
Tensor& hpu_wrap::cat_out(
    const TensorList tensors,
    int64_t dim_,
    Tensor& result) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "cat_out :",
      " tensprs=",
      to_string(tensors),
      " dim_=",
      to_string(dim_),
      " result=",
      to_string(result));

  FALLBACK_IF_UNSUPPORTED_OP(
      cat_out, PARAMS1(result, tensors[0]), PARAMS2(tensors, dim_, result))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return cat_hpu_lazy_out(result, tensors, dim_);

  } else {
    return cat_hpu_out(result, tensors, dim_);
  }
};
std::vector<Tensor> hpu_wrap::split_with_sizes(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "split_with_sizes :",
      " self=",
      to_string(self),
      " split_sizes=",
      to_string(split_sizes),
      " dim=",
      to_string(dim));
  FALLBACK_IF_UNSUPPORTED_OP(
      split_with_sizes, PARAMS1(self), PARAMS2(self, split_sizes, dim))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return split_with_sizes_hpu_lazy(self, split_sizes, dim);

  } else {
    return split_with_sizes_hpu(self, split_sizes, dim);
  }
};

std::tuple<Tensor, Tensor> hpu_wrap::sort(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "sort :",
      " self=",
      to_string(self),
      " dim =",
      to_string(dim),
      " descending=",
      to_string(descending));
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(dim), IValue(descending)};
  check_handle->hpu_check_ivalues("sort", op_stack);
  FALLBACK_IF_UNSUPPORTED_OP1_RT(
      self.scalar_type(), sort, PARAMS1(self), PARAMS2(self, dim, descending))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sort_hpu_lazy(self, dim, descending);

  } else {
    return sort_hpu(self, dim, descending);
  }
};

std::tuple<Tensor, Tensor, Tensor> hpu_wrap::_unique2(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "_unique2 :",
      " self=",
      to_string(self),
      " sorted=",
      to_string(sorted),
      " return_inverse=",
      to_string(return_inverse),
      " return_counts=",
      to_string(return_counts));
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self),
      IValue(sorted),
      IValue(return_inverse),
      IValue(return_counts)};
  check_handle->hpu_check_ivalues("unique", op_stack);

  FALLBACK_IF_UNSUPPORTED_OP1(
      _unique2,
      PARAMS1(self),
      PARAMS2(self, sorted, return_inverse, return_counts))

  if (sorted && self.dim() != 1) {
    FALLBACK_IF_UNSUPPORTED_OP2(
        _unique2, PARAMS2(self, sorted, return_inverse, return_counts))
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return unique2_hpu_lazy(self, sorted, return_inverse, return_counts);
  } else {
    return unique2_hpu(self, sorted, return_inverse, return_counts);
  }
}

std::vector<at::Tensor> hpu_wrap::unbind(const at::Tensor& self, int64_t dim) {
  PT_OP_TRACE;
  PT_OP_INFO("unbind :", " self=", to_string(self), " dim=", to_string(dim));
  FALLBACK_IF_UNSUPPORTED_OP_O(unbind, PARAMS1(self), PARAMS2(self, dim), int)

  return at::native::unbind(self, dim);
}

Tensor hpu_wrap::alias(const at::Tensor& self) {
  PT_OP_TRACE;
  PT_OP_INFO("alias :", " self=", to_string(self));
  return alias_hpu_lazy(self);
}

Tensor hpu_wrap::_unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "_unsafe_view:", " self=", to_string(self), " size=", to_string(size));
  FALLBACK_IF_UNSUPPORTED_OP(_unsafe_view, PARAMS1(self), PARAMS2(self, size))

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return view_hpu_lazy(self, size);

  } else {
    return view_hpu(self, size);
  }
}

at::Tensor hpu_wrap::squeeze(const at::Tensor& self) {
  PT_OP_TRACE;
  PT_OP_INFO("squeeze :", " self=", to_string(self));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    // using invalid dim size HABANA_DIM_MAX to signal the backend kernel that
    // squeeze needs to be performed on all applicable axes
    return squeeze_hpu_lazy(self, HABANA_DIM_MAX /*dim*/);
  } else {
    return at::native::squeeze(self);
  }
}

at::Tensor hpu_wrap::squeeze(const at::Tensor& self, int64_t dim) {
  PT_OP_TRACE;
  PT_OP_INFO("squeeze :", " self=", to_string(self), " dim=", to_string(dim));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return squeeze_hpu_lazy(self, dim);
  } else {
    return at::native::squeeze(self, dim);
  }
}

at::Tensor& hpu_wrap::squeeze_(at::Tensor& self) {
  PT_OP_TRACE;
  PT_OP_INFO("squeeze_ :", " self=", to_string(self));
  FALLBACK_IF_UNSUPPORTED_OP(squeeze_, PARAMS1(self), PARAMS2(self))

  return at::native::squeeze_(self);
}

at::Tensor& hpu_wrap::squeeze_(at::Tensor& self, int64_t dim) {
  PT_OP_TRACE;
  PT_OP_INFO("squeeze_ :", " self=", to_string(self), " dim=", to_string(dim));
  FALLBACK_IF_UNSUPPORTED_OP_O(squeeze_, PARAMS1(self), PARAMS2(self, dim), dim)

  return at::native::squeeze_(self, dim);
}

const at::Tensor& hpu_wrap::as_strided_(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  // No CPU fallback for as_strided_ since H2D & D2H DMA support only contiguous
  // tensor transfers
  // FALLBACK_IF_UNSUPPORTED_OP(__func__, PARAMS1(self)
  //  return AtenHpuTypeDefault::as_strided_(self, size, stride,
  //  storage_offset);
  PT_OP_TRACE;
  PT_OP_INFO(
      "as_strided_ :",
      " self=",
      to_string(self),
      " size=",
      to_string(size),
      " stride=",
      to_string(stride),
      " storage_offset=",
      to_string(storage_offset));
  return as_strided_hpu_lazy_(self, size, stride, storage_offset);
}

std::vector<at::Tensor> hpu_wrap::split(
    const at::Tensor& self,
    int64_t split_size,
    int64_t dim) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "split :",
      " self=",
      to_string(self),
      " split_size=",
      to_string(split_size),
      " dim=",
      to_string(dim));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      split, PARAMS1(self), PARAMS2(self, split_size, dim), Tensor)

  // lower aten::split as split_with_sizes using the logic used in Fork
  int64_t dim_size = self.size(dim);
  TORCH_CHECK(
      split_size > 0 || self.size(dim) == 0,
      "split_size can only be 0 if dimension size is 0, "
      "but got dimension size of ",
      dim_size);

  // if split_size is 0 and dimension size is 0, there is 1 split.
  int64_t num_splits = 1;
  if (split_size != 0) {
    // ensuring num_splits is at least 1 makes consistent the case where
    // split_size > dim_size (returns a single split).  We might want to error
    // here, but keep it for BC.
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }

  std::vector<int64_t> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = length;
  }

  IntArrayRef split_sizes(splits);
  return hpu_wrap::split_with_sizes(self, split_sizes, dim);
}

Tensor hpu_wrap::upsample_nearest2d(
    const Tensor& input,
    OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "upsample_nearest2d :",
      " input=",
      to_string(input),
      " output_size=",
      to_string(output_size),
      " scale_factors=",
      to_string(scale_factors));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      upsample_nearest2d,
      PARAMS1(input),
      PARAMS2(input, output_size, scale_factors),
      vec)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return upsample_nearest2d_hpu_lazy(input, output_size, scale_factors);
  } else {
    return upsample_nearest2d_hpu(input, output_size, scale_factors);
  }
};

Tensor hpu_wrap::upsample_nearest2d_backward(
    const Tensor& grad_output,
    OptionalIntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "upsample_nearest2d_backward :",
      " grad_output=",
      to_string(grad_output),
      " output_size=",
      to_string(output_size),
      " input_size=",
      to_string(input_size),
      " scale_factors=",
      to_string(scale_factors));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      upsample_nearest2d_backward,
      PARAMS1(grad_output),
      PARAMS2(grad_output, output_size, input_size, scale_factors),
      vec)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return upsample_nearest2d_backward_hpu_lazy(
        grad_output, output_size, input_size, scale_factors);
  } else {
    return upsample_nearest2d_backward_hpu(
        grad_output, output_size, input_size, scale_factors);
  }
};

Tensor hpu_wrap::upsample_nearest3d(
    const Tensor& input,
    OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "upsample_nearest3d :",
      " input=",
      to_string(input),
      " output_size=",
      to_string(output_size),
      " scale_factors=",
      to_string(scale_factors));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      upsample_nearest3d,
      PARAMS1(input),
      PARAMS2(input, output_size, scale_factors),
      vec)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return upsample_nearest3d_hpu_lazy(input, output_size, scale_factors);
  } else {
    return upsample_nearest3d_hpu(input, output_size, scale_factors);
  }
};

Tensor hpu_wrap::upsample_nearest3d_backward(
    const Tensor& grad_output,
    OptionalIntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "upsample_nearest3d_backward :",
      " grad_output=",
      to_string(grad_output),
      " output_size=",
      to_string(output_size),
      " input_size=",
      to_string(input_size),
      " scale_factors=",
      to_string(scale_factors));
  FALLBACK_IF_UNSUPPORTED_OP_O(
      upsample_nearest3d_backward,
      PARAMS1(grad_output),
      PARAMS2(grad_output, output_size, input_size, scale_factors),
      vec)

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return upsample_nearest3d_backward_hpu_lazy(
        grad_output, output_size, input_size, scale_factors);
  } else {
    return upsample_nearest3d_backward_hpu(
        grad_output, output_size, input_size, scale_factors);
  }
};

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_wrap(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor,
    float mom,
    bool nesterov) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "optimizer_sparse_sgd_with_valid_count :",
      " gradients=",
      to_string(gradients),
      " weights_in=",
      to_string(weights_in),
      " moments_in=",
      to_string(moments_in),
      " indices=",
      to_string(indices),
      " learning_rate=",
      to_string(learning_rate),
      " valid_count_tensor",
      to_string(valid_count_tensor),
      " mom",
      to_string(mom),
      " nesterov",
      to_string(nesterov));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return optimizer_sparse_sgd_with_valid_count_hpu_lazy(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor,
        mom,
        nesterov);
  } else {
    return optimizer_sparse_sgd_with_valid_count_hpu(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor,
        mom,
        nesterov);
  }
}
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "optimizer_sparse_adagrad_with_valid_count :",
      " gradients=",
      to_string(gradients),
      " weights_in=",
      to_string(weights_in),
      " moments_in=",
      to_string(moments_in),
      " indices=",
      to_string(indices),
      " learning_rate=",
      to_string(learning_rate),
      " valid_count_tensor",
      to_string(valid_count_tensor));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return optimizer_sparse_adagrad_with_valid_count_hpu_lazy(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor);
  } else {
    return optimizer_sparse_adagrad_with_valid_count_hpu(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor);
  }
}
void optimizer_adamw_hpu_wrap(
    const TensorList& gradient_vec,
    TensorList& weight_vec,
    TensorList& exp_avg_vec,
    TensorList& exp_avg_sq_vec,
    const float lr,
    at::Tensor& neg_step_t,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "optimizer_adamw :",
      " gradient_vec=",
      to_string(gradient_vec),
      " weight_vec=",
      to_string(weight_vec),
      " exp_avg_vec=",
      to_string(exp_avg_vec),
      " exp_avg_sq_vec=",
      to_string(exp_avg_sq_vec),
      " lr=",
      to_string(lr),
      " neg_step_t",
      to_string(neg_step_t),
      " beta1",
      to_string(beta1),
      " beta2",
      to_string(beta2),
      " epsilon",
      to_string(epsilon),
      " weight_decay",
      to_string(weight_decay));
  TORCH_CHECK((weight_vec.size() > 0), "Can not process empty weight vector");
  auto lr_t = get_tensor_for_scalar(lr);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_adamw_hpu_lazy(
        gradient_vec,
        weight_vec,
        exp_avg_vec,
        exp_avg_sq_vec,
        lr_t,
        neg_step_t,
        beta1,
        beta2,
        epsilon,
        weight_decay);
  } else {
    optimizer_adamw_hpu(
        gradient_vec,
        weight_vec,
        exp_avg_vec,
        exp_avg_sq_vec,
        lr_t,
        neg_step_t,
        beta1,
        beta2,
        epsilon,
        weight_decay);
  }
}
Tensor fused_norm_hpu_wrap(
    std::vector<at::Tensor>& grad,
    const Tensor& max_norm,
    float norm_type) {
  PT_OP_TRACE;
  PT_OP_INFO(
      "fused_norm :",
      " grad=",
      to_string(grad),
      " max_norm=",
      to_string(max_norm),
      " norm_type=",
      to_string(norm_type));
  TORCH_CHECK((grad.size() > 0), "Can not process empty grad vector");
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return fused_norm_hpu_lazy(grad, max_norm, norm_type);
  } else {
    return fused_norm_hpu(grad, max_norm, norm_type);
  }
}
void optimizer_adagrad_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " optimizer_adagrad:",
      " gradients=",
      to_string(gradients),
      " weights=",
      to_string(weights),
      " variances=",
      to_string(variances),
      " epoch_num=",
      to_string(epoch_num),
      " lr=",
      to_string(lr),
      " wd=",
      to_string(wd),
      " lrd=",
      to_string(lrd),
      " epsilon=",
      to_string(epsilon));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_adagrad_hpu_lazy(
        gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
  } else {
    optimizer_adagrad_hpu(
        gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
  }
}

void optimizer_ema_hpu_wrap(
    const TensorList& model_inputs,
    TensorList& updated_ema,
    const at::Tensor& decay) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " optimizer_ema:",
      " model_inputs=",
      to_string(model_inputs),
      " updated_ema=",
      to_string(updated_ema),
      " decay=",
      to_string(decay));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_ema_hpu_lazy(model_inputs, updated_ema, decay);
  }

  return;
}

void optimizer_sgd_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " optimizer_sgd:",
      " gradients=",
      to_string(gradients),
      " weights=",
      to_string(weights),
      " lr=",
      to_string(lr),
      " wd=",
      to_string(wd),
      " mom=",
      to_string(mom),
      " damp=",
      to_string(damp),
      " nesterov=",
      to_string(nesterov));
  if (!habana_lazy::isDeviceInLoweringMode() &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_sgd_hpu_lazy(gradients, weights, lr, wd, mom, damp, nesterov);
  } else {
    optimizer_sgd_hpu(gradients, weights, lr, wd, mom, damp, nesterov);
  }
}

void optimizer_sgd_momentum_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " optimizer_sgd_momentum:",
      " gradients=",
      to_string(gradients),
      " weights=",
      to_string(weights),
      " momentum=",
      to_string(momentum),
      " epoch_num=",
      to_string(epoch_num),
      " lr=",
      to_string(lr),
      " wd=",
      to_string(wd),
      " mom=",
      to_string(mom),
      " damp=",
      to_string(damp),
      " nesterov=",
      to_string(nesterov));
  auto mom_t = get_tensor_for_scalar(mom);
  if (!habana_lazy::isDeviceInLoweringMode() &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_sgd_momentum_hpu_lazy(
        gradients, weights, momentum, epoch_num, lr, mom_t, wd, damp, nesterov);
  } else {
    optimizer_sgd_momentum_hpu(
        gradients, weights, momentum, epoch_num, lr, mom_t, wd, damp, nesterov);
  }
}

Tensor optimizer_lamb_fused_norm_hpu_wrap(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " optimizer_lamb_fused_norm:",
      " grad=",
      to_string(grad),
      "max_grad_norm=",
      to_string(max_grad_norm));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return optimizer_lamb_fused_norm_hpu_lazy(grad, max_grad_norm);
  } else {
    return optimizer_lamb_fused_norm_hpu(grad, max_grad_norm);
  }
}

std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
optimizer_lamb_phase1_hpu_wrap(
    const std::vector<at::Tensor>& gradients,
    std::vector<at::Tensor>& weights,
    std::vector<at::Tensor>& exp_avg,
    std::vector<at::Tensor>& exp_avg_sq,
    const at::Tensor& clip_global_grad_norm,
    const int grad_averaging,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " optimizer_lamb_phase1:",
      " gradients=",
      to_string(gradients),
      "weights=",
      to_string(weights),
      "exp_avg=",
      to_string(exp_avg),
      "exp_avg_sq=",
      to_string(exp_avg_sq),
      "clip_global_grad_norm=",
      to_string(clip_global_grad_norm),
      "grad_averaging=",
      to_string(grad_averaging),
      "lr=",
      to_string(lr),
      "beta1=",
      to_string(beta1),
      "beta2=",
      to_string(beta2),
      "epsilon=",
      to_string(epsilon),
      "step=",
      to_string(step),
      "bias_correction=",
      to_string(bias_correction),
      "weight_decay=",
      to_string(weight_decay));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return optimizer_lamb_phase1_hpu_lazy(
        gradients,
        weights,
        exp_avg,
        exp_avg_sq,
        clip_global_grad_norm,
        grad_averaging,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        bias_correction,
        weight_decay);
  } else {
    return optimizer_lamb_phase1_hpu(
        gradients,
        weights,
        exp_avg,
        exp_avg_sq,
        clip_global_grad_norm,
        grad_averaging,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        bias_correction,
        weight_decay);
  }
}

void optimizer_lamb_phase2_hpu_wrap(
    std::vector<at::Tensor>& weight_vec,
    const std::vector<at::Tensor>& adam_norm_vec,
    const std::vector<at::Tensor>& weight_norm_vec,
    const std::vector<at::Tensor>& adam_step_vec,
    const float step,
    const float weight_decay,
    const int use_lamb) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " optimizer_lamb_phase2:",
      " weight_vec=",
      to_string(weight_vec),
      "adam_norm_vec=",
      to_string(adam_norm_vec),
      "weight_norm_vec=",
      to_string(weight_norm_vec),
      "adam_step_vec=",
      to_string(adam_step_vec),
      "step=",
      to_string(step),
      "weight_decay=",
      to_string(weight_decay),
      "use_lamb=",
      to_string(use_lamb));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_lamb_phase2_hpu_lazy(
        weight_vec,
        adam_norm_vec,
        weight_norm_vec,
        adam_step_vec,
        step,
        weight_decay,
        use_lamb);
  } else {
    optimizer_lamb_phase2_hpu(
        weight_vec,
        adam_norm_vec,
        weight_norm_vec,
        adam_step_vec,
        step,
        weight_decay,
        use_lamb);
  }
}

void optimizer_lars_hpu_wrap(
    const at::TensorList& params,
    at::TensorList& grads,
    const std::vector<int64_t> skipMasks,
    const float eeta,
    const float weight_decay,
    const float eps,
    const float lr) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " optimizer_lars_hpu_wrap:",
      " param=",
      to_string(params),
      " grad=",
      to_string(grads),
      " skipMasks=",
      to_string(skipMasks),
      "eeta=",
      to_string(eeta),
      "weight_decay=",
      to_string(weight_decay),
      "eps=",
      to_string(eps),
      "lr=",
      to_string(lr));
  return optimizer_lars_hpu_lazy(
      params, grads, skipMasks, eeta, weight_decay, eps, lr);
}

void optimizer_ResourceApplyMomentum_hpu_wrap(
    at::TensorList& params_momentum_buffer_list,
    const at::TensorList& d_p_list,
    const float momentum) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " optimizer_ResourceApplyMomentum_hpu_wrap:",
      " params_momentum_buffer_list =",
      to_string(params_momentum_buffer_list),
      " d_p_list=",
      to_string(d_p_list),
      "momentum=",
      to_string(momentum));
  return optimizer_ResourceApplyMomentum_hpu_lazy(
      params_momentum_buffer_list, d_p_list, momentum);
}

Tensor torchvision_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    double iou_threshold) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " torchvision_nms:",
      " boxes=",
      to_string(boxes),
      "scores=",
      to_string(scores),
      "iou_threshold=",
      to_string(iou_threshold));

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return habana_nms_hpu_lazy(
        boxes, scores, iou_threshold, -std::numeric_limits<float>::max());
  } else {
    return habana_nms_hpu(
        boxes, scores, iou_threshold, -std::numeric_limits<float>::max());
  }
}

Tensor habana_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    float iou_threshold,
    float score_threshold) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " habana_nms:",
      " boxes=",
      to_string(boxes),
      "scores=",
      to_string(scores),
      "iou_threshold=",
      to_string(iou_threshold),
      "score_threshold=",
      to_string(score_threshold));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return habana_nms_hpu_lazy(boxes, scores, iou_threshold, score_threshold);
  } else {
    return habana_nms_hpu(boxes, scores, iou_threshold, score_threshold);
  }
}

Tensor batched_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    const at::Tensor& indices,
    float iou_threshold) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " batched_nms:",
      " boxes=",
      to_string(boxes),
      "scores=",
      to_string(scores),
      "indices=",
      to_string(indices),
      "iou_threshold=",
      to_string(iou_threshold));
  return batched_nms_hpu_lazy(boxes, scores, indices, iou_threshold);
}

Tensor hpu_wrap::_masked_scale(
    const Tensor& self,
    const Tensor& mask,
    double scale) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " _masked_scale:",
      " self=",
      to_string(self),
      "mask=",
      to_string(mask),
      "scale=",
      to_string(scale));
  FALLBACK_IF_UNSUPPORTED_OP(
      _masked_scale, PARAMS1(self, mask), PARAMS2(self, mask, scale))
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return masked_scale_hpu_lazy(self, mask, scale);
  } else {
    return masked_scale_hpu(self, mask, scale);
  }
}

Tensor habana_d2d_memcpy(const Tensor& self) {
  HABANA_ASSERT(0);
  return self;
}

Tensor habana_d2d_memcpy_other(const Tensor& self, Tensor& other) {
  HABANA_ASSERT(0);
  static_cast<void>(other);
  return self;
}

/***********************************************************************************
 * Kernels requiring autograd override
 **********************************************************************************/
using namespace torch::autograd;

// Pytorch fork's isfinite is a compound op that is realized through a
// sequence of simpler ops. For better performance, using the underlying TPC
// kernel.

struct IsfiniteFunction : public torch::autograd::Function<IsfiniteFunction> {
  static at::Tensor forward(AutogradContext* ctx, at::Tensor input) {
    at::Tensor result;
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
      result = isfinite_hpu_lazy(input);
    } else {
      result = isfinite_hpu(input);
    }

    static_cast<void>(ctx);
    return result;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    static_cast<void>(ctx);
    static_cast<void>(grad_output);
    return {};
  }
};

Tensor hpu_wrap::isfinite(const Tensor& self) {
  PT_OP_TRACE;
  PT_OP_INFO(" isfinite:", " self=", to_string(self));
  return IsfiniteFunction::apply(self);
}

struct MatmulFunction : public torch::autograd::Function<MatmulFunction> {
  static at::Tensor forward(
      AutogradContext* ctx,
      const at::Tensor& self,
      const at::Tensor& other) {
    at::Tensor result;
    ctx->save_for_backward({self, other});
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
      result = matmul_hpu_lazy(self, other);
    } else {
      result = matmul_hpu(self, other);
    }

    return result;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    std::tuple<Tensor, Tensor> result;
    variable_list saved_vars = ctx->get_saved_variables();

    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
      result = matmul_backward_hpu_lazy(
          grad_output[0], saved_vars[0], saved_vars[1]);
    } else {
      result =
          matmul_backward_hpu(grad_output[0], saved_vars[0], saved_vars[1]);
    }

    return {std::get<0>(result), std::get<1>(result)};
  }
};

Tensor hpu_wrap::matmul(const Tensor& self, const Tensor& other) {
  PT_OP_TRACE;
  PT_OP_INFO("matmul:", " self=", to_string(self), "other=", to_string(other));
  return MatmulFunction::apply(self, other);
};

struct LinearFunction : public torch::autograd::Function<LinearFunction> {
  static at::Tensor forward(
      AutogradContext* ctx,
      Tensor input,
      Tensor weight,
      c10::optional<Tensor> bias_opt) {
    at::Tensor result;
    // ctx->save_for_backward<> does not take c10::optional<Tensor> bias_opt
    // So create and use an "undefined" tensor if bias_opt does not have value
    auto bias = bias_opt.value_or(Tensor());
    ctx->save_for_backward({input, weight, bias});
    result = linear_(input, weight, bias_opt);
    return result;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    std::tuple<Tensor, Tensor> result;
    variable_list saved_vars = ctx->get_saved_variables();
    Tensor input = saved_vars[0];
    Tensor weight = saved_vars[1];
    Tensor bias_opt = saved_vars[2];
    return linear_non2d_bwd_hpu_lazy(grad_output[0], input, weight, bias_opt);
  }
};

Tensor hpu_wrap::linear(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  PT_KERNEL_DEBUG(
      "HpuOp linear:",
      " input=",
      to_string(input),
      " weight=",
      to_string(weight),
      " bias_opt=",
      to_string(bias_opt));
  return LinearFunction::apply(input, weight, bias_opt);
}

struct AdaptiveAvgPool2DFunction
    : public torch::autograd::Function<AdaptiveAvgPool2DFunction> {
  static at::Tensor forward(
      AutogradContext* ctx,
      const Tensor& input,
      IntArrayRef output_size) {
    ctx->save_for_backward({input});
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
      return adaptive_avg_pool2d_hpu_lazy(input, output_size);
    } else {
      return adaptive_avg_pool2d_hpu(input, output_size);
    }
  }
  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    variable_list saved_vars = ctx->get_saved_variables();
    auto& input = saved_vars[0];
    at::Tensor result;
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
      result = adaptive_avg_pool2d_backward_hpu_lazy(grad_output[0], input);
    } else {
      result = adaptive_avg_pool2d_backward_hpu(grad_output[0], input);
    }
    return {result, torch::Tensor()};
  }
};
Tensor hpu_wrap::adaptive_avg_pool2d(
    const Tensor& input,
    IntArrayRef output_size) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " adaptive_avg_pool2d:",
      " input=",
      to_string(input),
      "output_size=",
      to_string(output_size));
  return AdaptiveAvgPool2DFunction::apply(input, output_size);
};

Tensor hpu_wrap::slice(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " slice:",
      " self=",
      to_string(self),
      "dim=",
      to_string(dim),
      "start=",
      to_string(start),
      "end=",
      to_string(end),
      "step=",
      to_string(step));

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return slice_hpu_lazy(self, dim, start, end, step);
  } else {
    return slice_hpu(self, dim, start, end, step);
  }
};

struct DropoutFunction : public Function<DropoutFunction> {
  static at::Tensor forward(
      AutogradContext* ctx,
      at::Tensor input,
      double p,
      bool train) {
    ctx->saved_data["p"] = p;
    if ((p == 0) || !train || (input.numel() == 0)) {
      return input;
    } else if (p == 1) {
      return input * 0.0;
    }
    c10::optional<at::Generator> gen = c10::nullopt;
    at::Tensor result1, result2;
    std::tie(result1, result2) = _fused_dropout(input, p, gen);
    ctx->save_for_backward({result2});
    return result1;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    auto p = ctx->saved_data["p"].toDouble();
    if (p == 0) {
      return {grad_output[0], torch::Tensor(), torch::Tensor()};
    } else if (p == 1) {
      return {grad_output[0] * 0.0, torch::Tensor(), torch::Tensor()};
    }
    variable_list saved_vars = ctx->get_saved_variables();
    auto mask = saved_vars[0];
    at::Tensor result;
    result = hpu_wrap::_masked_scale(grad_output[0], mask, 1.0 / p);
    return {result, torch::Tensor(), torch::Tensor()};
  }
};

Tensor hpu_wrap::dropout(const Tensor& input, double p, bool train) {
  PT_OP_TRACE;
  PT_OP_INFO(
      " dropout:",
      " input=",
      to_string(input),
      "p=",
      to_string(p),
      "train=",
      to_string(train));
  return DropoutFunction::apply(input, p, train);
}

namespace vision {
namespace ops {
at::Tensor roi_align_fwd_wrap(
    const at::Tensor& images,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t output_h,
    int64_t output_w,
    int64_t sampling_ratio,
    bool aligned) {
  int mode = 0;
  // rois from torchvision are of shape {K, 5} where 1st column contain the
  // index of corresponding element in the batch, whereas remaining columns
  // contain the roi co-ordinates. Since "roi_align" TPC kernels expect these
  // indices and co-ordinates as separate tensors, therefore split operation is
  // being done here. TPC kernel expects a 1D Int tensor for num_rois, therefore
  // a reshape and conversion to Int is also done here.
  auto out = rois.split_with_sizes({1, 4}, 1);
  auto num_rois = out[0].view(-1).to(torch::kInt);
  auto roi = out[1];
  return roi_align_fwd_hpu_lazy(
      images,
      roi,
      num_rois,
      static_cast<int>(output_h),
      static_cast<int>(output_w),
      mode,
      static_cast<int>(sampling_ratio),
      static_cast<float>(spatial_scale),
      aligned);
};

at::Tensor roi_align_bwd_wrap(
    const at::Tensor& grad_out,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t output_h,
    int64_t output_w,
    int64_t bs,
    int64_t ch,
    int64_t h,
    int64_t w,
    int64_t sampling_ratio,
    bool aligned) {
  static_cast<void>(output_h);
  static_cast<void>(output_w);
  // Refer to comment in roi_align_fwd_wrap for same operations
  auto out = rois.split_with_sizes({1, 4}, 1);
  auto num_rois = out[0].view(-1).to(torch::kInt);
  auto roi = out[1];
  return roi_align_bwd_hpu_lazy(
      grad_out,
      roi,
      num_rois,
      static_cast<int>(bs),
      static_cast<int>(ch),
      static_cast<int>(h),
      static_cast<int>(w),
      static_cast<int>(sampling_ratio),
      static_cast<float>(spatial_scale),
      aligned);
};

TORCH_LIBRARY_IMPL(torchvision, HPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      TORCH_FN(roi_align_fwd_wrap));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_align_backward"),
      TORCH_FN(roi_align_bwd_wrap));
}
} // namespace ops
} // namespace vision

TORCH_LIBRARY(hpu, m) {
  m.def("cat(Tensor[] tensors, int dim, Tensor out_shape) -> Tensor");
  m.def(
      "repeat_inlv(Tensor input, Tensor repeats, int dim, Tensor out_shape) -> Tensor");
  m.def(
      "nonzero(Tensor self, Tensor? nonzero_input_shape_tensor) -> (Tensor, Tensor)");
  m.def(
      "index_put(Tensor self, Tensor where_tensor, Tensor shape_tensor, Tensor value, Tensor value_upd_dim, Tensor zero_shape_tensor, bool accumulate=False) -> Tensor");
  m.def("mul_out(Tensor self, Tensor other, Tensor(a!) out) -> Tensor(a!)");
  m.def("div_out(Tensor self, Tensor other, Tensor(a!) out) -> Tensor(a!)");
  m.def("mm_t(Tensor mm, Tensor t , bool tr, bool no_tr) -> Tensor");
  m.def("habana_d2d_memcpy_other(Tensor s, Tensor(a!) d) -> Tensor(a!)");
  m.def(
      "sum_dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def(
      "prod_dim_Int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("all_dim(Tensor self, int dim, bool keepdim=False) -> Tensor");
  m.def(
      "arange_out(Scalar start, Scalar end, Scalar step, Tensor(a!) out) -> Tensor(a!)");
  m.def("arange_out_ds(Tensor shape, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "arange_out_ds_ht(Tensor host, Tensor(a!) out, Tensor out_shape) -> Tensor(a!)");
  m.def("diag_out(Tensor self, int diagonal, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "randperm_out(int n, Generator? generator, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "randperm_out_ds(Tensor idst, Generator? generator, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "randperm_out_ds_ht(Tensor ht, Tensor st, Generator? generator, Tensor output) -> Tensor(a!)");
  m.def(
      "max_dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
  m.def("habana_d2d_memcpy(Tensor self) -> Tensor");
  m.def(
      "habanaOptimizerSparseSgd(Tensor gradients, Tensor(a!) weights_in, Tensor(b!) moments_in, Tensor indices, Tensor learning_rate, Tensor valid_count_tensor, float mom, bool nesterov) -> (Tensor(a!), Tensor(b!))");
  m.def(
      "habanaOptimizerSparseAdagrad(Tensor gradients, Tensor(a!) weights_in, Tensor(b!) moments_in, Tensor indices, Tensor learning_rate, Tensor valid_count_tensor) -> (Tensor(a!), Tensor(b!))");
  m.def("cast(Tensor self, Scalar type) -> Tensor(a)");
  m.def(
      "embedding_bag_sum(Tensor input, Tensor indices, Tensor offsets, Tensor valid_count, int kernel_mode) -> Tensor");
  m.def(
      "embedding_bag_sum_bwd_out(Tensor(a!) out, Tensor input, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, int kernel_mode) -> Tensor(a!)");
  m.def(
      "habanaOptimizerFusedAdagrad(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!)[] variances_in, Tensor epoch_num, Tensor(c!) learning_rate, float wd, float lrd, float eps) -> ()");
  m.def(
      "habanaOptimizerFusedSGD(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!) learning_rate, float wd, float mom, float damp, bool nesterov) -> ()");
  m.def(
      "habanaOptimizerFusedSGDMomentum(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!)[] momentum_in, Tensor epoch_num, Tensor(c!) learning_rate, Tensor mom, float wd, float damp, bool nesterov) -> ()");
  m.def(
      "hpu::habanaOptimizerAdamW(Tensor[] gradient_vec, Tensor(a!)[] weight_vec, Tensor(b!)[] exp_avg_vec, Tensor(c!)[] exp_avg_sq_vec, Tensor(d!) lr_t, Tensor(e!) neg_step_t, float beta1, float beta2, float epsilon, float weight_decay) -> ()");
  m.def(
      "hpu::habanaOptimizerFusedEMA(Tensor[] model_inputs, Tensor(a!)[] updated_ema, Tensor decay) -> ()");
  m.def(
      "fused_norm_(Tensor(a!)[] grad, Tensor max_norm, float norm_type) -> Tensor");
  m.def(
      "fused_norm_lazy(Tensor(a!)[] grad, Tensor max_norm, float norm_type) -> Tensor");
  m.def(
      "habanaOptimizerLambFusedNorm(Tensor[] grad, float max_norm, Tensor clip_norm) -> Tensor");
  m.def(
      "habanaOptimizerLambPhase1(Tensor[] grad, Tensor[] weights, Tensor[] exp_avg, Tensor[] exp_avg_sq, Tensor clip_global_grad_norm, float beta1, float beta2, float beta3, float epsilon, Tensor bias_corection1, Tensor bias_correction2, float weight_decay) -> (Tensor[], Tensor[], Tensor[])");
  m.def(
      "habanaOptimizerLambPhase2(Tensor(a!)[] weights, Tensor[] adam_norm, Tensor[] wt_norm, Tensor[] adam_step, Tensor neg_step, float wd, int use_lamb) -> ()");
  m.def(
      "habanaOptimizerLars(Tensor[] params, Tensor(a!)[] grads, Tensor lr_t, int[] skip_masks, float eeta, float weight_decay, float eps) -> ()");
  m.def(
      "habanaOptimizerResourceApplyMomentum(Tensor(a!)[] params_momentum_buf_list, Tensor[] dp_list, float momentum) -> ()");
  m.def(
      "habana_nms(Tensor boxes, Tensor scores, float iou_threshold, float score_threshold) -> (Tensor, Tensor, Tensor)");
  m.def(
      "batched_nms(Tensor boxes, Tensor scores, Tensor indexes, float iou_threshold, Tensor shape_tensor1, Tensor shape_tensor2, int max_classes) -> (Tensor, Tensor)");
  m.def(
      "roi_align_fwd(Tensor inputs, Tensor rois, Tensor n_rois, int out_h, int out_w, int mode, int sr, float ss, bool aligned) -> Tensor");
  m.def(
      "roi_align_bwd(Tensor inputs, Tensor rois, Tensor n_rois, Tensor input_shape, int sr, float ss, bool aligned) -> Tensor");
  m.def(
      "_unique(Tensor self, bool sorted, bool return_inverse) -> (Tensor, Tensor)");
  m.def(
      "_unique2(Tensor self, bool sorted, bool return_inverse, bool return_counts) -> (Tensor, Tensor)");
  m.def(
      "unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor)");
  m.def(
      "gather_elements(Tensor self, Tensor index, Tensor? opt, int64_t dim_, bool sorted) -> Tensor");
  m.def("permute(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("permute_cl(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("restride_cl(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("restride(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("permute_weight(Tensor self, int[] size) -> (Tensor)");
  m.def("permuted_weight_restride(Tensor self, int[] size) -> (Tensor)");
  m.def("control_edge_other_(Tensor self, Tensor(a) other) -> Tensor(a)");
  m.def("control_edge_(Tensor(a) self)-> Tensor(a)");
  m.def(
      "hpu::native_batch_norm_training(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::native_batch_norm_inf(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor)");
  m.def(
      "hpu::native_batch_norm_backward(Tensor input, Tensor? grad_out, Tensor? weight, Tensor? mean, Tensor? invistd, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "as_strided_lazy_(Tensor self, int[] size, int[] stride, int offset, bool can_replace) -> (Tensor)");
  m.def(
      "as_strided_lazy_cl_(Tensor self, int[] size, int[] stride, int offset, bool can_replace) -> (Tensor)");
  m.def(
      "strided_view(Tensor self, int[] size, int[] stride, int offset) -> (Tensor)");
  m.def(
      "strided_view_out(Tensor self, int[] size, int[] stride, int offset) -> (Tensor)");
  m.def(
      "strided_view_cl(Tensor self, int[] size, int[] stride, int offset) -> (Tensor)");
  m.def("strided_view_ds(Tensor self, Tensor size, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_out_ds(Tensor self, Tensor size, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_cl_ds(Tensor self, Tensor size,Tensor offset) -> (Tensor)");
  m.def("slice_insert(Tensor self, Tensor other, int[] params) -> (Tensor)");
  m.def(
      "strided_insert(Tensor self, Tensor other, int[] stride, int offset) -> (Tensor)");
  m.def(
      "strided_insert_cl(Tensor self, Tensor other, int[] stride, int offset) -> (Tensor)");
  m.def(
      "strided_insert_ds(Tensor self, Tensor other, Tensor offset) -> (Tensor)");
  m.def(
      "strided_insert_cl_ds(Tensor self, Tensor other, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_orig_ds(Tensor self, Tensor size, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_out_orig_ds(Tensor self, Tensor size, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_insert_orig_ds(Tensor self, Tensor other, Tensor stride, Tensor offset) -> (Tensor)");
  m.def("as_strided_layout(Tensor self, int[] size) -> (Tensor)");
  m.def("reshape(Tensor self, int[] size) -> (Tensor)");
  m.def(
      "matmul_backward(Tensor grad_out, Tensor self, Tensor other) -> (Tensor, Tensor)");
  m.def(
      "instance_norm(Tensor input, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "instance_norm_backward(Tensor input, Tensor grad_in, Tensor? mean, Tensor? istd, Tensor gamma) -> (Tensor, Tensor, Tensor)");
  m.def("view(Tensor input, Tensor shape) -> Tensor");
  m.def(
      "slice(Tensor input, Tensor shape, Tensor step,  Tensor start) -> (Tensor)");
  m.def(
      "hpu::expand(Tensor(a) self, Tensor shape, *, bool implicit=False) -> Tensor(a)");
  m.def("hpu::repeat(Tensor self, Tensor repeats_shape) -> Tensor");
  m.def(
      "hpu::constant_pad_nd(Tensor self, Tensor pad_before_tensor, Tensor pad_after_tensor, Scalar value) -> Tensor");
  m.def(
      "hpu::repeat_ht(Tensor self, Tensor repeats_shape, Tensor result_shape) -> Tensor");
  m.def(
      "hpu::constant_pad_nd_ht(Tensor self, Tensor pad_tensor, Tensor output_shape_tensor, Scalar value) -> Tensor");
  m.def(
      "upsample_nearest2d_backward(Tensor grad_output, int[]? output_size, Tensor input_size, float[]? scale_factors) -> Tensor");
  m.def(
      "hpu::topk(Tensor self, Tensor k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)");
  m.def(
      "hpu::scatter_nd_onnx(Tensor input, Tensor indices, Tensor values) -> Tensor");
  m.def(
      "hpu::scatter_nd(Tensor input, Tensor indices, Tensor grouped_indices, Tensor update_locations, Tensor updates) -> Tensor");
  m.def(
      "hpu::_fused_dropout(Tensor input, float p, Tensor? seed) -> (Tensor, Tensor)");
  m.def(
      "hpu::linear_non2d_bwd(Tensor grad_out, Tensor input, Tensor weight, Tensor? bias) -> (Tensor, Tensor, Tensor)");
  m.def("hpu::identity(Tensor self) -> (Tensor)");
}

TORCH_LIBRARY_IMPL(torchvision, HPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::nms"),
      TORCH_FN(torchvision_nms_hpu_wrap));
}

TORCH_LIBRARY(hccl, m) {
  m.def(
      "broadcast_(Tensor(a!) tensor, int root_rank, int64_t comm_id) -> Tensor(a!)");
  m.def(
      "allreduce_(Tensor(a!) tensor, uint8_t reduceOp, int64_t comm_id) -> Tensor(a!)");
  m.def(
      "reduce_(Tensor(a!) tensor, int64_t dst_rank, uint8_t reduceOp, int64_t comm_id) -> Tensor(a!)");
  m.def(
      "alltoall_out(Tensor input_tensor, int64_t comm_id, Tensor(a!) output_tensor) -> Tensor(a!)");
  m.def(
      "allgather_out(Tensor input_tensor, int64_t comm_id, Tensor(a!) output_tensor) -> Tensor(a!)");
  m.def(
      "reduce_scatter_out(Tensor input_tensor, uint8_t reduceOp, int64_t comm_id, Tensor(a!) output_tensor) -> Tensor(a!)");
  m.def(
      "send_(Tensor(a!) tensor,  int64_t dst_rank,  int64_t tag, int64_t comm_id) -> Tensor(a!)");
  m.def(
      "recv_(Tensor(a!) tensor,  int64_t src_rank,  int64_t tag, int64_t comm_id) -> Tensor(a!)");
}
