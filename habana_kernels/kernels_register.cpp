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

#include "habana_kernels/aten_hpu_type_default.h"
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/lazy_executor.h"
#include "kernel_input_checks.h"
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
  return is_pinned_hpu(self, device);
}

Tensor hpu_wrap::pin_memory(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  return pin_memory_hpu(self, device);
};

Tensor hpu_wrap::_pin_memory(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  return pin_memory_hpu(self, device);
};

Tensor& hpu_wrap::copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  if (src.device().type() == c10::DeviceType::HPU &&
      self.device().type() == c10::DeviceType::HPU) {
    if (src.scalar_type() == c10::ScalarType::Float &&
        self.scalar_type() == c10::ScalarType::Byte) {
      return habana::AtenHpuTypeDefault::copy_(self, src, non_blocking);
    }
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return copy_hpu_lazy_(self, src, non_blocking);
  } else {
    if (src.device().type() == c10::DeviceType::HPU &&
        self.device().type() == c10::DeviceType::HPU) {
      if (src.scalar_type() == c10::ScalarType::Float &&
          self.scalar_type() == c10::ScalarType::Long) {
        return habana::AtenHpuTypeDefault::copy_(self, src, non_blocking);
      }
    }
    return copy_hpu_(self, src, non_blocking);
  }
};

Tensor hpu_wrap::_reshape_alias(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride) {
  if (!hpu_check_inputs_impl("_reshape_alias", {self}))
    return AtenHpuTypeDefault::_reshape_alias(self, size, stride);
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
  // if (!hpu_check_inputs_impl("as_strided", {self}))
  //  return AtenHpuTypeDefault::as_strided(self, size, stride, storage_offset);

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
  if (!hpu_check_inputs_impl("set_", {self}))
    return AtenHpuTypeDefault::set_(self, source, storage_offset, size, stride);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return set_hpu_lazy_(self, source, storage_offset, size, stride);

  } else {
    return set_hpu_(self, source, storage_offset, size, stride);
  }
};

Tensor hpu_wrap::view(const Tensor& self, IntArrayRef size) {
  if (!hpu_check_inputs_impl("view", {self}))
    return AtenHpuTypeDefault::view(self, size);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return view_hpu_lazy(self, size);

  } else {
    return view_hpu(self, size);
  }
};

Tensor hpu_wrap::addcmul(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& alpha) {
  if (!hpu_check_inputs_impl("addcmul", {self, tensor1, tensor2}))
    return AtenHpuTypeDefault::addcmul(self, tensor1, tensor2, alpha);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return addcmul_hpu_lazy(self, tensor1, tensor2, alpha);

  } else {
    return addcmul_hpu(self, tensor1, tensor2, alpha);
  }
};
Tensor& hpu_wrap::addcmul_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& alpha) {
  if (!hpu_check_inputs_impl("addcmul_", {self, tensor1, tensor2}))
    return AtenHpuTypeDefault::addcmul_(self, tensor1, tensor2, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return addcmul_hpu_lazy_(self, tensor1, tensor2, alpha);

  } else {
    return addcmul_hpu_(self, tensor1, tensor2, alpha);
  }
};
Tensor hpu_wrap::addcdiv(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& alpha) {
  if (!hpu_check_inputs_impl("addcdiv", {self, tensor1, tensor2}))
    return AtenHpuTypeDefault::addcdiv(self, tensor1, tensor2, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return addcdiv_hpu_lazy(self, tensor1, tensor2, alpha);

  } else {
    return addcdiv_hpu(self, tensor1, tensor2, alpha);
  }
};
Tensor& hpu_wrap::addcdiv_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& alpha) {
  if (!hpu_check_inputs_impl("addcdiv_", {self, tensor1, tensor2}))
    return AtenHpuTypeDefault::addcdiv_(self, tensor1, tensor2, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return addcdiv_hpu_lazy_(self, tensor1, tensor2, alpha);

  } else {
    return addcdiv_hpu_(self, tensor1, tensor2, alpha);
  }
};
Tensor hpu_wrap::add(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  if (!hpu_check_inputs_impl("add", {self, other}))
    return AtenHpuTypeDefault::add(self, other, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return add_tensor_hpu_lazy(self, other, alpha);
  } else {
    return add_tensor_hpu(self, other, alpha);
  }
};
Tensor hpu_wrap::add(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  if (!hpu_check_inputs_impl("add", {self}))
    return AtenHpuTypeDefault::add(self, other, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return add_scalar_hpu_lazy(self, other, alpha);

  } else {
    return add_scalar_hpu(self, other, alpha);
  }
};
Tensor& hpu_wrap::add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  if (!hpu_check_inputs_impl("add_", {self}))
    return AtenHpuTypeDefault::add_(self, other, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return add_scalar_hpu_lazy_(self, other, alpha);

  } else {
    return add_scalar_hpu_(self, other, alpha);
  }
};
Tensor& hpu_wrap::add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  if (!hpu_check_inputs_impl("add_", {self, other}))
    return AtenHpuTypeDefault::add_(self, other, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return add_tensor_hpu_lazy_(self, other, alpha);
  } else {
    return add_tensor_hpu_(self, other, alpha);
  }
};
Tensor hpu_wrap::sub(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  if (!hpu_check_inputs_impl("sub", {self, other}))
    return AtenHpuTypeDefault::sub(self, other, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sub_tensor_hpu_lazy(self, other, alpha);

  } else {
    return sub_tensor_hpu(self, other, alpha);
  }
};
Tensor& hpu_wrap::sub_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  if (!hpu_check_inputs_impl("sub_", {self, other}))
    return AtenHpuTypeDefault::sub_(self, other, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sub_tensor_hpu_lazy_(self, other, alpha);

  } else {
    return sub_tensor_hpu_(self, other, alpha);
  }
};
Tensor hpu_wrap::sub(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  if (!hpu_check_inputs_impl("sub", {self}))
    return AtenHpuTypeDefault::sub(self, other, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sub_scalar_hpu_lazy(self, other, alpha);

  } else {
    return sub_scalar_hpu(self, other, alpha);
  }
};
Tensor& hpu_wrap::sub_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  if (!hpu_check_inputs_impl("sub_", {self}))
    return AtenHpuTypeDefault::sub_(self, other, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sub_scalar_hpu_lazy_(self, other, alpha);

  } else {
    return sub_scalar_hpu_(self, other, alpha);
  }
};
Tensor hpu_wrap::rsub(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  if (!hpu_check_inputs_impl("rsub", {self}))
    return AtenHpuTypeDefault::rsub(self, other, alpha);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return rsub_scalar_hpu_lazy(self, other, alpha);

  } else {
    return rsub_scalar_hpu(self, other, alpha);
  }
};
Tensor hpu_wrap::_s_where(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return where_tensor_hpu_lazy(condition, self, other);
  } else {
    return where_tensor_hpu(condition, self, other);
  }
}
Tensor& hpu_wrap::mul_(Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("mul_", {self, other}))
    return AtenHpuTypeDefault::mul_(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mul_tensor_hpu_lazy_(self, other);
  } else {
    return mul_tensor_hpu_(self, other);
  }
};
Tensor hpu_wrap::mul(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("mul", {self, other}))
    return AtenHpuTypeDefault::mul(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mul_tensor_hpu_lazy(self, other);
  } else {
    return mul_tensor_hpu(self, other);
  }
};

Tensor& hpu_wrap::mul_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  if (!hpu_check_inputs_impl("mul_out", {out, self, other}))
    return AtenHpuTypeDefault::mul_out(self, other, out);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mul_out_hpu_lazy(out, self, other);
  } else {
    return mul_out_hpu(out, self, other);
  }
};

Tensor hpu_wrap::mul(const Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("mul", {self}))
    return AtenHpuTypeDefault::mul(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mul_scalar_hpu_lazy(self, other);

  } else {
    return mul_scalar_hpu(self, other);
  }
};
Tensor& hpu_wrap::mul_(Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("mul_", {self}))
    return AtenHpuTypeDefault::mul_(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mul_scalar_hpu_lazy_(self, other);

  } else {
    return mul_scalar_hpu_(self, other);
  }
};
Tensor hpu_wrap::div(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("div", {self, other}))
    return AtenHpuTypeDefault::div(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return div_tensor_hpu_lazy(self, other);

  } else {
    return div_tensor_hpu(self, other);
  }
};
Tensor& hpu_wrap::div_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  if (!hpu_check_inputs_impl("div_out", {result, self, other}))
    return AtenHpuTypeDefault::div_out(self, other, result);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return div_tensor_hpu_lazy_out(result, self, other);

  } else {
    return div_tensor_hpu_out(result, self, other);
  }
};
Tensor& hpu_wrap::div_(Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("div_", {self, other}))
    return AtenHpuTypeDefault::div_(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return div_tensor_hpu_lazy_(self, other);

  } else {
    return div_tensor_hpu_(self, other);
  }
};
Tensor hpu_wrap::div(const Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("div", {self}))
    return AtenHpuTypeDefault::div(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return div_scalar_hpu_lazy(self, other);

  } else {
    return div_scalar_hpu(self, other);
  }
};
Tensor& hpu_wrap::div_(Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("div_", {self}))
    return AtenHpuTypeDefault::div_(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return div_scalar_hpu_lazy_(self, other);

  } else {
    return div_scalar_hpu_(self, other);
  }
};

Tensor hpu_wrap::floor_divide(const Tensor& self, const Tensor& other) {
  TORCH_WARN_ONCE(
      "floor_divide is deprecated, and will be removed in a future version of pytorch. "
      "It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). "
      "This results in incorrect rounding for negative values.\n"
      "To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), "
      "or for actual floor division, use torch.div(a, b, rounding_mode='floor').");
  if (!hpu_check_inputs_impl("div", {self, other}))
    return AtenHpuTypeDefault::floor_divide(self, other);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return floor_divide_tensor_hpu_lazy(self, other);
  } else {
    HABANA_ASSERT(
        0 && "floor_divide with rounding mode not implemented for eager mode");
    return div_tensor_hpu(self, other);
  }
};

Tensor hpu_wrap::remainder(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("remainder", {self, other}))
    return AtenHpuTypeDefault::remainder(self, other);
  return remainder_tensor_hpu(self, other);
};

Tensor& hpu_wrap::remainder_(Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("remainder_", {self, other}))
    return AtenHpuTypeDefault::remainder_(self, other);
  return remainder_tensor_hpu_(self, other);
};

Tensor hpu_wrap::remainder(const Tensor& self, const at::Scalar& other) {
  if (!hpu_check_inputs_impl("remainder", {self}))
    return AtenHpuTypeDefault::remainder(self, other);
  return remainder_scalar_hpu(self, other);
};

Tensor& hpu_wrap::remainder_(Tensor& self, const at::Scalar& other) {
  if (!hpu_check_inputs_impl("remainder_", {self}))
    return AtenHpuTypeDefault::remainder_(self, other);
  return remainder_scalar_hpu_(self, other);
};

Tensor& hpu_wrap::remainder_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  if (!hpu_check_inputs_impl("remainder_out", {self, other, result}))
    return AtenHpuTypeDefault::remainder_out(self, other, result);
  return remainder_tensor_hpu_out(self, other, result);
};

Tensor& hpu_wrap::remainder_out(
    const Tensor& self,
    const at::Scalar& other,
    Tensor& result) {
  if (!hpu_check_inputs_impl("remainder_out", {self, result}))
    return AtenHpuTypeDefault::remainder_out(self, other, result);
  return remainder_scalar_hpu_out(self, other, result);
};

Tensor& hpu_wrap::diag_out(const Tensor& self, int64_t diagonal, Tensor& out) {
  if (!hpu_check_inputs_impl("diag_out", {self, out}))
    return AtenHpuTypeDefault::diag_out(self, diagonal, out);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return diag_hpu_lazy_out(self, diagonal, out);
  } else {
    return diag_hpu_out(self, diagonal, out);
  }
}

Tensor hpu_wrap::pow(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("pow", {self, other}))
    return AtenHpuTypeDefault::pow(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return pow_tensor_tensor_hpu_lazy(self, other);

  } else {
    return pow_tensor_tensor_hpu(self, other);
  }
};
Tensor& hpu_wrap::pow_(Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("pow_", {self, other}))
    return AtenHpuTypeDefault::pow_(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return pow_tensor_tensor_hpu_lazy_(self, other);

  } else {
    return pow_tensor_tensor_hpu_(self, other);
  }
};
Tensor hpu_wrap::pow(const Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("pow", {self}))
    return AtenHpuTypeDefault::pow(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return pow_tensor_scalar_hpu_lazy(self, other);

  } else {
    return pow_tensor_scalar_hpu(self, other);
  }
};
Tensor& hpu_wrap::pow_(Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("pow_", {self}))
    return AtenHpuTypeDefault::pow_(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return pow_tensor_scalar_hpu_lazy_(self, other);

  } else {
    return pow_tensor_scalar_hpu_(self, other);
  }
};
Tensor hpu_wrap::pow(const Scalar& other, const Tensor& self) {
  if (!hpu_check_inputs_impl("pow", {self}))
    return AtenHpuTypeDefault::pow(other, self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return pow_scalar_tensor_hpu_lazy(other, self);

  } else {
    return pow_scalar_tensor_hpu(other, self);
  }
};

Tensor hpu_wrap::maximum(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("maximum", {self, other}))
    return AtenHpuTypeDefault::maximum(self, other);
  return maximum_hpu(self, other);
};

Tensor hpu_wrap::minimum(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("minimum", {self, other}))
    return AtenHpuTypeDefault::minimum(self, other);
  return minimum_hpu(self, other);
};

Tensor hpu_wrap::gt(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("gt", {self, other}))
    return AtenHpuTypeDefault::gt(self, other);

  return gt_tensor_hpu(self, other);
};

Tensor hpu_wrap::gt(const Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("gt", {self}))
    return AtenHpuTypeDefault::gt(self, other);

  return gt_scalar_hpu(self, other);
};
Tensor& hpu_wrap::eq_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  if (!hpu_check_inputs_impl("eq_out", {output, self, other}))
    return AtenHpuTypeDefault::eq_out(self, other, output);

  return eq_tensor_out_hpu(output, self, other);
};
Tensor hpu_wrap::eq(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("eq", {self, other}))
    return AtenHpuTypeDefault::eq(self, other);

  return eq_tensor_hpu(self, other);
};
Tensor hpu_wrap::eq(const Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("eq", {self}))
    return AtenHpuTypeDefault::eq(self, other);

  return eq_tensor_scalar_hpu(self, other);
};
Tensor hpu_wrap::lt(const Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("lt", {self}))
    return AtenHpuTypeDefault::lt(self, other);

  return lt_scalar_hpu(self, other);
};
Tensor hpu_wrap::lt(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("lt", {self, other}))
    return AtenHpuTypeDefault::lt(self, other);

  return lt_tensor_hpu(self, other);
};
Tensor hpu_wrap::ge(const Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("ge", {self}))
    return AtenHpuTypeDefault::ge(self, other);

  return ge_scalar_hpu(self, other);
};
Tensor hpu_wrap::ge(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("ge", {self, other}))
    return AtenHpuTypeDefault::ge(self, other);

  return ge_tensor_hpu(self, other);
};
Tensor hpu_wrap::le(const Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("le", {self}))
    return AtenHpuTypeDefault::le(self, other);

  return le_scalar_hpu(self, other);
};
Tensor hpu_wrap::le(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("le", {self, other}))
    return AtenHpuTypeDefault::le(self, other);

  return le_tensor_hpu(self, other);
};
Tensor hpu_wrap::ne(const Tensor& self, const Scalar& other) {
  if (!hpu_check_inputs_impl("ne", {self}))
    return AtenHpuTypeDefault::ne(self, other);
  /* This code will be exercised only in legacy eager mode. The following code
   * is used to replace the graph transformation based ne op implementation */
  Scalar c = 0.0;
  Tensor e = hpu_wrap::eq(self, other);
  return hpu_wrap::eq(e, c);
};
Tensor hpu_wrap::ne(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("ne", {self, other}))
    return AtenHpuTypeDefault::ne(self, other);
  /* This code will be exercised only in legacy eager mode. The following code
   * is used to replace the graph transformation based ne op implementation */
  Scalar c = 0.0;
  Tensor e = hpu_wrap::eq(self, other);
  return hpu_wrap::eq(e, c);
};

Tensor& hpu_wrap::all_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& out) {
  if (!hpu_check_inputs_impl("all_out", {self, out}))
    return AtenHpuTypeDefault::all_out(self, dim, keepdim, out);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return all_dim_out_hpu_lazy(self, dim, keepdim, out);
  } else {
    HABANA_ASSERT(0 && "all_out is not implemented for eager mode");
    return all_dim_out_hpu_lazy(self, dim, keepdim, out);
  }
};

Tensor hpu_wrap::all(const Tensor& self, int64_t dim, bool keepdim) {
  if (!hpu_check_inputs_impl("all", {self}))
    return AtenHpuTypeDefault::all(self, dim, keepdim);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return all_dim_hpu_lazy(self, dim, keepdim);
  } else {
    return all_dim_hpu(self, dim, keepdim);
  }
};
Tensor hpu_wrap::all(const Tensor& self) {
  if (!hpu_check_inputs_impl("all", {self}))
    return AtenHpuTypeDefault::all(self);
  return all_hpu(self);
};
Tensor hpu_wrap::convolution_overrideable(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  auto bias = bias_opt.value_or(Tensor());
  if (!hpu_check_inputs_impl("convolution_overrideable", {input, weight, bias}))
    return AtenHpuTypeDefault::convolution_overrideable(
        input,
        weight,
        bias_opt,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return convolution_hpu_lazy(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups);
  } else {
    return convolution_hpu(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups);
  }
};
std::tuple<Tensor, Tensor, Tensor> hpu_wrap::convolution_backward_overrideable(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  if (!hpu_check_inputs_impl(
          "convolution_backward_overrideable", {grad_output, input, weight}))
    return AtenHpuTypeDefault::convolution_backward_overrideable(
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return convolution_backward_hpu_lazy(
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask);

  } else {
    return convolution_backward_hpu(
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask);
  }
};

Tensor hpu_wrap::constant_pad_nd(
    const Tensor& self,
    IntArrayRef pad,
    const Scalar& value) {
  if (!hpu_check_inputs_impl("constant_pad_nd", {self}))
    return AtenHpuTypeDefault::constant_pad_nd(self, pad, value);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return constant_pad_hpu_lazy(self, pad, value);

  } else {
    return constant_pad_hpu(self, pad, value);
  }
};
Tensor hpu_wrap::embedding(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(weight),
      IValue(indices),
      IValue(padding_idx),
      IValue(scale_grad_by_freq),
      IValue(sparse)};
  check_handle->hpu_check_ivalues("embedding", op_stack);
  if (!(hpu_check_inputs_impl("embedding", {weight, indices}) &&
        (check_handle->get_status()))) {
    return AtenHpuTypeDefault::embedding(
        weight, indices, padding_idx, scale_grad_by_freq, sparse);
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return embedding_hpu_lazy(
        weight, indices, padding_idx, scale_grad_by_freq, sparse);

  } else {
    return embedding_hpu(
        weight, indices, padding_idx, scale_grad_by_freq, sparse);
  }
};
Tensor hpu_wrap::embedding_dense_backward(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(grad),
      IValue(indices),
      IValue(num_weights),
      IValue(padding_idx),
      IValue(scale_grad_by_freq)};
  check_handle->hpu_check_ivalues("embedding_dense_backward", op_stack);
  if (!(hpu_check_inputs_impl("embedding_dense_backward", {grad, indices}) &&
        (check_handle->get_status()))) {
    return AtenHpuTypeDefault::embedding_dense_backward(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return embedding_dense_backward_hpu_lazy(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);

  } else {
    return embedding_dense_backward_hpu(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
};
Tensor embedding_bag_sum_hpu_wrap(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
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
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return embedding_bag_sum_bwd_out_kernel_mode_hpu_lazy(
        out, input, indices, offsets, valid_count, kernel_mode);

  } else {
    return embedding_bag_sum_bwd_out_kernel_mode_hpu(
        out, input, indices, offsets, valid_count, kernel_mode);
  }
};
Tensor& hpu_wrap::fill_(Tensor& self, const Scalar& value) {
  if (!hpu_check_inputs_impl("fill_", {self}))
    return AtenHpuTypeDefault::fill_(self, value);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return fill_hpu_lazy_(self, value);
  } else {
    return fill_hpu_(self, value);
  }
};
Tensor& hpu_wrap::masked_fill_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  if (!hpu_check_inputs_impl("masked_fill_", {self, mask, value}))
    return AtenHpuTypeDefault::masked_fill_(self, mask, value);

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
  if (!hpu_check_inputs_impl("masked_fill_", {self, mask}))
    return AtenHpuTypeDefault::masked_fill_(self, mask, value);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return masked_fill_scalar_hpu_lazy_(self, mask, value);

  } else {
    return masked_fill_scalar_hpu_(self, mask, value);
  }
};
Tensor hpu_wrap::masked_select(const Tensor& self, const Tensor& mask) {
  if (!hpu_check_inputs_impl("masked_select", {self, mask}))
    return AtenHpuTypeDefault::masked_select(self, mask);

  std::vector<Tensor> idx;

  if (mask.dim() == 0) {
    idx = mask.unsqueeze(0).nonzero().unbind(1);
  } else {
    idx = mask.nonzero().unbind(1);
  }

  c10::List<c10::optional<Tensor>> converted_inds;
  converted_inds.reserve(idx.size());
  for (size_t i = 0; i < idx.size(); ++i) {
    const auto& ind = idx[i];
    if (ind.defined()) {
      converted_inds.push_back(ind.to(ind.options().device("hpu")));
    } else {
      converted_inds.push_back(std::move(idx[i]));
    }
  }
  return hpu_wrap::index(self, converted_inds);
};
Tensor hpu_wrap::gather(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(dim_), IValue(index), IValue(sparse_grad)};
  check_handle->hpu_check_ivalues("gather_elements", op_stack);
  if (!(hpu_check_inputs_impl("gather_elements", {self, index}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::gather(self, dim_, index, sparse_grad);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return gather_src_hpu_lazy(self, dim_, index, sparse_grad);

  } else {
    return gather_src_hpu(self, dim_, index, sparse_grad);
  }
};
Tensor& hpu_wrap::scatter_(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  if (!hpu_check_inputs_impl("scatter_", {self, index, src}))
    return AtenHpuTypeDefault::scatter_(self, dim_, index, src);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return scatter_inplace_src_hpu_lazy(self, dim_, index, src);

  } else {
    return scatter_inplace_src_hpu(self, dim_, index, src);
  }
};
Tensor hpu_wrap::scatter(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  if (!hpu_check_inputs_impl("scatter", {self, index, src}))
    return AtenHpuTypeDefault::scatter(self, dim_, index, src);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return scatter_src_hpu_lazy(self, dim_, index, src);

  } else {
    return scatter_src_hpu(self, dim_, index, src);
  }
};
Tensor& hpu_wrap::scatter_(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Scalar& value) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return scatter_inplace_value_hpu_lazy(self, dim_, index, value);

  } else {
    return scatter_inplace_value_hpu(self, dim_, index, value);
  }
};
Tensor hpu_wrap::scatter_add(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  if (!hpu_check_inputs_impl("scatter_add", {self, index, src}))
    return AtenHpuTypeDefault::scatter_add(self, dim_, index, src);

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
  if (!hpu_check_inputs_impl("scatter_add_", {self, index, src}))
    return AtenHpuTypeDefault::scatter_add_(self, dim_, index, src);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return scatter_add_inplace_src_hpu_lazy(self, dim_, index, src);

  } else {
    return scatter_add_inplace_src_hpu(self, dim_, index, src);
  }
};
Tensor& hpu_wrap::index_add_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source,
    const Scalar& alpha) {
  Tensor alpha_times_source = hpu_wrap::mul(source, alpha);
  if (!hpu_check_inputs_impl("index_add_", {self, indices, source}))
    return AtenHpuTypeDefault::index_add_(
        self, dim_, indices, alpha_times_source);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return index_add_hpu_lazy_(self, dim_, indices, alpha_times_source);

  } else {
    return index_add_hpu_(self, dim_, indices, alpha_times_source);
  }
};
Tensor& hpu_wrap::index_add_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source) {
  if (!hpu_check_inputs_impl("index_add_", {self, indices, source}))
    return AtenHpuTypeDefault::index_add_(self, dim_, indices, source);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return index_add_hpu_lazy_(self, dim_, indices, source);

  } else {
    return index_add_hpu_(self, dim_, indices, source);
  }
};
Tensor hpu_wrap::index_put(
    const Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    bool accumulate) {
  if (!hpu_check_inputs_impl(
          "index_put", {self, indices[0].value_or(Tensor()), value}))
    return AtenHpuTypeDefault::index_put(self, indices, value, accumulate);
  // TODO: Need a better way to handle this rather than converting everywhere
  std::vector<at::Tensor> indices_list;
  for (const c10::optional<Tensor>& input : indices) {
    indices_list.push_back(input.value_or(Tensor()));
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return index_put_hpu_lazy(
        self, at::TensorList(indices_list), value, accumulate);

  } else {
    return index_put_hpu(self, at::TensorList(indices_list), value, accumulate);
  }
};
Tensor& hpu_wrap::index_put_(
    Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    bool accumulate) {
  if (!hpu_check_inputs_impl(
          "index_put_", {self, indices[0].value_or(Tensor()), value}))
    return AtenHpuTypeDefault::index_put_(self, indices, value, accumulate);

  std::vector<at::Tensor> indices_list;
  for (const c10::optional<Tensor>& input : indices) {
    indices_list.push_back(input.value_or(Tensor()));
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return index_put_hpu_lazy_(self, indices_list, value, accumulate);

  } else {
    return index_put_hpu_(self, indices_list, value, accumulate);
  }
};
Tensor hpu_wrap::index(
    const at::Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices) {
  if (!hpu_check_inputs_impl("index", {self, indices[0].value_or(Tensor())}))
    return AtenHpuTypeDefault::index(self, indices);

  // we dont support AdvanceIndexing where index tensor is empty
  // for now fallback to cpu, will add once we get tpc index kernel
  std::vector<at::Tensor> indices_list;
  for (const c10::optional<Tensor>& input : indices) {
    if (input.has_value() && !input->defined()) {
      return AtenHpuTypeDefault::index(self, indices);
    } else {
      indices_list.push_back(input.value());
    }
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return index_hpu_lazy(self, indices_list);
  } else {
    return index_hpu(self, indices_list);
  }
};

Tensor& hpu_wrap::_index_put_impl_(
    Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  if (!hpu_check_inputs_impl(
          "_index_put_impl_", {self, indices[0].value_or(Tensor())}))
    return AtenHpuTypeDefault::_index_put_impl_(
        self, indices, value, accumulate, unsafe);

  std::vector<at::Tensor> indices_list;
  for (const c10::optional<Tensor>& input : indices) {
    indices_list.push_back(input.value_or(Tensor()));
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return _index_put_impl_hpu_lazy_(
        self, indices_list, value, accumulate, unsafe);
  } else {
    return AtenHpuTypeDefault::_index_put_impl_(
        self, indices, value, accumulate, unsafe);
  }
}

Tensor hpu_wrap::index_select(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  if (!hpu_check_inputs_impl("index_select", {self, index}))
    return AtenHpuTypeDefault::index_select(self, dim, index);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return index_select_hpu_lazy(self, dim, index);

  } else {
    return index_select_hpu(self, dim, index);
  }
};
Tensor gather2d_hpu_wrap(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return gather2d_hpu_lazy(input, indices, validCount);

  } else {
    return gather2d_hpu(input, indices, validCount);
  }
};
Tensor hpu_wrap::select(const Tensor& self, int64_t dim, int64_t index) {
  if (!hpu_check_inputs_impl("select", {self}))
    return AtenHpuTypeDefault::select(self, dim, index);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return select_hpu_lazy(self, dim, index);

  } else {
    return select_hpu(self, dim, index);
  }
};
Tensor& hpu_wrap::arange_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& output) {
  if (!hpu_check_inputs_impl("arange_out", {output}))
    return AtenHpuTypeDefault::arange_out(start, end, step, output);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return arange_hpu_lazy(output, start, end, step);

  } else {
    return arange_hpu(output, start, end, step);
  }
};
Tensor hpu_wrap::nonzero(const Tensor& self) {
  if (!hpu_check_inputs_impl("nonzero", {self}))
    return AtenHpuTypeDefault::nonzero(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return nonzero_hpu_lazy(self);
  } else {
    return nonzero_hpu(self);
  }
};
Tensor hpu_wrap::mm(const at::Tensor& mat1, const at::Tensor& mat2) {
  if (!hpu_check_inputs_impl("mm", {mat1, mat2}))
    return AtenHpuTypeDefault::mm(mat1, mat2);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mm_hpu_lazy(mat1, mat2);
  } else {
    return mm_hpu(mat1, mat2);
  }
};

Tensor hpu_wrap::baddbmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  Tensor out = hpu_wrap::mul(hpu_wrap::bmm(mat1, mat2), alpha);
  if (beta.toFloat() != 0) {
    hpu_wrap::add_(out, self, beta);
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
  if (beta.toFloat() == 0) {
    hpu_wrap::bmm_out(mat1, mat2, out);
    hpu_wrap::mul_(out, alpha);
  } else {
    Tensor r_bmul = hpu_wrap::mul(self, beta);
    hpu_wrap::bmm_out(mat1, mat2, out);
    hpu_wrap::mul_(out, alpha);
    hpu_wrap::add_(out, r_bmul, 1);
  }
  return out;
}

Tensor& hpu_wrap::baddbmm_(
    Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  if (beta.toFloat() == 0) {
    hpu_wrap::bmm_out(mat1, mat2, self);
    hpu_wrap::mul_(self, alpha);
  } else {
    Tensor r_bmm = hpu_wrap::bmm(mat1, mat2);
    hpu_wrap::mul_(self, beta);
    hpu_wrap::add_(self, r_bmm, alpha);
  }
  return self;
}

Tensor hpu_wrap::addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(mat1), IValue(mat2), IValue(beta), IValue(alpha)};
  check_handle->hpu_check_ivalues("addmm", op_stack);
  if (!(hpu_check_inputs_impl("addmm", {self, mat1, mat2}) &&
        check_handle->get_status())) {
    return AtenHpuTypeDefault::addmm(self, mat1, mat2, beta, alpha);
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return addmm_hpu_lazy(self, mat1, mat2, beta, alpha);
  } else {
    return addmm_hpu(self, mat1, mat2, beta, alpha);
  }
};
Tensor& hpu_wrap::bmm_out(const Tensor& self, const Tensor& mat2, Tensor& out) {
  if (!hpu_check_inputs_impl("bmm_out", {out, self, mat2}))
    return AtenHpuTypeDefault::bmm_out(self, mat2, out);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return batch_gemm_out_hpu_lazy(out, self, mat2);

  } else {
    return batch_gemm_out_hpu(out, self, mat2);
  }
};
Tensor hpu_wrap::bmm(const Tensor& self, const Tensor& mat2) {
  if (!hpu_check_inputs_impl("bmm", {self, mat2}))
    return AtenHpuTypeDefault::bmm(self, mat2);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return batch_gemm_hpu_lazy(self, mat2);

  } else {
    return batch_gemm_hpu(self, mat2);
  }
};
Tensor hpu_wrap::dot(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("dot", {self, other}))
    return AtenHpuTypeDefault::dot(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return dot_hpu_lazy(self, other);

  } else {
    return dot_hpu(self, other);
  }
};
Tensor hpu_wrap::mv(const Tensor& self, const Tensor& other) {
  if (!hpu_check_inputs_impl("mv", {self, other}))
    return AtenHpuTypeDefault::mv(self, other);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mv_hpu_lazy(self, other);

  } else {
    return mv_hpu(self, other);
  }
};
std::tuple<Tensor, Tensor> hpu_wrap::nll_loss_forward(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  auto weight = weight_opt.value_or(Tensor());
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self),
      IValue(target),
      IValue(weight_opt),
      IValue(reduction),
      IValue(ignore_index)};
  // PyTorch or user may send a "undefined" tensor, since there is no way of
  // comparing "undefined" tensors for equality, therefore evaluate for defined
  // and send bool to attribute based cpu fallback logic.
  if (weight_opt.has_value()) {
    op_stack.erase(op_stack.cbegin() + 2);
    op_stack.insert(
        op_stack.cbegin() + 2, IValue(weight_opt.value().defined()));
  }
  check_handle->hpu_check_ivalues("nll_loss_forward", op_stack);
  if (!(hpu_check_inputs_impl("nll_loss_forward", {self, target, weight}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::nll_loss_forward(
        self, target, weight_opt, reduction, ignore_index);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return nll_loss_forward_hpu_lazy(
        self, target, weight, reduction, ignore_index);

  } else {
    return nll_loss_forward_hpu(self, target, weight, reduction, ignore_index);
  }
};
Tensor hpu_wrap::nll_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    UNUSED const Tensor& total_weight) {
  auto weight = weight_opt.value_or(Tensor());
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(grad_output),
      IValue(self),
      IValue(target),
      IValue(weight_opt),
      IValue(reduction),
      IValue(ignore_index),
      IValue(total_weight)};
  // PyTorch or user may send a "undefined" tensor, since there is no way of
  // comparing "undefined" tensors for equality, therefore evaluate for defined
  // and send bool to attribute based cpu fallback logic.
  if (weight_opt.has_value()) {
    op_stack.erase(op_stack.cbegin() + 3);
    op_stack.insert(
        op_stack.cbegin() + 3, IValue(weight_opt.value().defined()));
  }
  check_handle->hpu_check_ivalues("nll_loss_backward", op_stack);
  if (!(hpu_check_inputs_impl(
            "nll_loss_backward",
            {grad_output, self, target, weight, total_weight}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::nll_loss_backward(
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return nll_loss_backward_hpu_lazy(
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight);

  } else {
    return nll_loss_backward_hpu(
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight);
  }
};

Tensor hpu_wrap::nll_loss2d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    UNUSED const Tensor& total_weight) {
  auto weight = weight_opt.value_or(Tensor());
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(grad_output),
      IValue(self),
      IValue(target),
      IValue(weight_opt),
      IValue(reduction),
      IValue(ignore_index),
      IValue(total_weight)};
  // PyTorch or user may send a "undefined" tensor, since there is no way of
  // comparing "undefined" tensors for equality, therefore evaluate for defined
  // and send bool to attribute based cpu fallback logic.
  if (weight_opt.has_value()) {
    op_stack.erase(op_stack.cbegin() + 3);
    op_stack.insert(
        op_stack.cbegin() + 3, IValue(weight_opt.value().defined()));
  }
  check_handle->hpu_check_ivalues("nll_loss2d_backward", op_stack);
  if (!(hpu_check_inputs_impl(
            "nll_loss2d_backward",
            {grad_output, self, target, weight, total_weight}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::nll_loss2d_backward(
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return nll_loss2d_backward_hpu_lazy(
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight);

  } else {
    return nll_loss2d_backward_hpu(
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight);
  }
};

std::tuple<Tensor, Tensor> hpu_wrap::nll_loss2d_forward(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  auto weight = weight_opt.value_or(Tensor());
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self),
      IValue(target),
      IValue(weight_opt),
      IValue(reduction),
      IValue(ignore_index)};
  // PyTorch or user may send a "undefined" tensor, since there is no way of
  // comparing "undefined" tensors for equality, therefore evaluate for defined
  // and send bool to attribute based cpu fallback logic.
  if (weight_opt.has_value()) {
    op_stack.erase(op_stack.cbegin() + 2);
    op_stack.insert(
        op_stack.cbegin() + 2, IValue(weight_opt.value().defined()));
  }
  check_handle->hpu_check_ivalues("nll_loss2d_forward", op_stack);
  if (!(hpu_check_inputs_impl("nll_loss2d_forward", {self, target, weight}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::nll_loss2d_forward(
        self, target, weight, reduction, ignore_index);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return nll_loss2d_forward_hpu_lazy(
        self, target, weight, reduction, ignore_index);

  } else {
    return nll_loss2d_forward_hpu(
        self, target, weight, reduction, ignore_index);
  }
};

Tensor hpu_wrap::mse_loss(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  if (!hpu_check_inputs_impl("mse_loss", {self, target}))
    return AtenHpuTypeDefault::mse_loss(self, target, reduction);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mse_loss_forward_hpu_lazy(self, target, reduction);
  } else {
    return mse_loss_forward_hpu(self, target, reduction);
  }
};
Tensor hpu_wrap::mse_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  if (!hpu_check_inputs_impl("mse_loss_backward", {grad_output, self, target}))
    return AtenHpuTypeDefault::mse_loss_backward(
        grad_output, self, target, reduction);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mse_loss_backward_hpu_lazy(grad_output, self, target, reduction);

  } else {
    return mse_loss_backward_hpu(grad_output, self, target, reduction);
  }
};

Tensor hpu_wrap::kl_div(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(target), IValue(reduction), IValue(log_target)};
  check_handle->hpu_check_ivalues("kl_div", op_stack);
  if (!(hpu_check_inputs_impl("kl_div", {self, target}) &&
        check_handle->get_status())) {
    return AtenHpuTypeDefault::kl_div(self, target, reduction, log_target);
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return kl_div_hpu_lazy(self, target, reduction, log_target);
  } else {
    return kl_div_hpu(self, target, reduction, log_target);
  }
};

Tensor hpu_wrap::kl_div_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(grad),
      IValue(self),
      IValue(target),
      IValue(reduction),
      IValue(log_target)};
  check_handle->hpu_check_ivalues("kl_div_backward", op_stack);
  if (!(hpu_check_inputs_impl("kl_div_backward", {grad, self, target}) &&
        (check_handle->get_status()))) {
    return AtenHpuTypeDefault::kl_div_backward(
        grad, self, target, reduction, log_target);
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return kl_div_backward_hpu_lazy(grad, self, target, reduction, log_target);
  } else {
    return kl_div_backward_hpu(grad, self, target, reduction, log_target);
  }
};

Tensor hpu_wrap::binary_cross_entropy(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction) {
  auto weight = weight_opt.value_or(Tensor());
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(target), IValue(weight_opt), IValue(reduction)};
  // PyTorch or user may send a "undefined" tensor, since there is no way of
  // comparing "undefined" tensors for equality, therefore evaluate for defined
  // and send bool to attribute based cpu fallback logic.
  if (weight_opt.has_value()) {
    op_stack.erase(op_stack.cbegin() + 2);
    op_stack.insert(
        op_stack.cbegin() + 2, IValue(weight_opt.value().defined()));
  }
  check_handle->hpu_check_ivalues("binary_cross_entropy", op_stack);
  if (!(hpu_check_inputs_impl("binary_cross_entropy", {self, target, weight}) &&
        check_handle->get_status())) {
    return AtenHpuTypeDefault::binary_cross_entropy(
        self, target, weight, reduction);
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return binary_cross_entropy_hpu_lazy(self, target, weight, reduction);

  } else {
    return binary_cross_entropy_hpu(self, target, weight, reduction);
  }
};
Tensor hpu_wrap::binary_cross_entropy_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction) {
  auto weight = weight_opt.value_or(Tensor());
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(grad_output),
      IValue(self),
      IValue(target),
      IValue(weight_opt),
      IValue(reduction)};
  // PyTorch or user may send a "undefined" tensor, since there is no way of
  // comparing "undefined" tensors for equality, therefore evaluate for defined
  // and send bool to attribute based cpu fallback logic.
  if (weight_opt.has_value()) {
    op_stack.erase(op_stack.cbegin() + 3);
    op_stack.insert(
        op_stack.cbegin() + 3, IValue(weight_opt.value().defined()));
  }
  check_handle->hpu_check_ivalues("binary_cross_entropy_backward", op_stack);
  if (!(hpu_check_inputs_impl(
            "binary_cross_entropy_backward",
            {grad_output, self, target, weight}) &&
        (check_handle->get_status()))) {
    return AtenHpuTypeDefault::binary_cross_entropy_backward(
        grad_output, self, target, weight, reduction);
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return binary_cross_entropy_backward_hpu_lazy(
        grad_output, self, target, weight, reduction);

  } else {
    return binary_cross_entropy_backward_hpu(
        grad_output, self, target, weight, reduction);
  }
};
Tensor hpu_wrap::binary_cross_entropy_with_logits(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& pos_weight,
    int64_t reduction) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self),
      IValue(target),
      IValue(weight),
      IValue(pos_weight),
      IValue(reduction)};
  // PyTorch or user may send a "undefined" tensor, since there is no way of
  // comparing "undefined" tensors for equality, therefore evaluate for defined
  // and send bool to attribute based cpu fallback logic.
  if (weight.has_value()) {
    op_stack.erase(op_stack.cbegin() + 2);
    op_stack.insert(op_stack.cbegin() + 2, IValue(weight.value().defined()));
  }
  if (pos_weight.has_value()) {
    op_stack.erase(op_stack.cbegin() + 3);
    op_stack.insert(
        op_stack.cbegin() + 3, IValue(pos_weight.value().defined()));
  }
  check_handle->hpu_check_ivalues("binary_cross_entropy_with_logits", op_stack);
  if (!(hpu_check_inputs_impl(
            "binary_cross_entropy_with_logits",
            {self,
             target,
             weight.value_or(Tensor()),
             pos_weight.value_or(Tensor())}) &&
        (check_handle->get_status()))) {
    return AtenHpuTypeDefault::binary_cross_entropy_with_logits(
        self, target, weight, pos_weight, reduction);
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return binary_cross_entropy_with_logits_hpu_lazy(
        self, target, weight, pos_weight, reduction);
  } else {
    return binary_cross_entropy_with_logits_hpu(
        self, target, weight, pos_weight, reduction);
  }
};
std::tuple<Tensor, Tensor, Tensor> hpu_wrap::native_batch_norm(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps) {
  auto weight = weight_opt.value();
  auto bias = bias_opt.value();
  auto running_mean = running_mean_opt.value();
  auto running_var = running_var_opt.value();

  if (!hpu_check_inputs_impl(
          "native_batch_norm",
          {input, weight, bias, running_mean, running_var}))
    return AtenHpuTypeDefault::native_batch_norm(
        input,
        weight_opt,
        bias_opt,
        running_mean_opt,
        running_var_opt,
        training,
        momentum,
        eps);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return batch_norm_hpu_lazy(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps);

  } else {
    return batch_norm_hpu(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps);
  }
};
std::tuple<Tensor, Tensor, Tensor> hpu_wrap::native_batch_norm_backward(
    const Tensor& grad_out,
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    const c10::optional<Tensor>& save_mean_opt,
    const c10::optional<Tensor>& save_invstd_opt,
    bool train,
    double eps,
    std::array<bool, 3> output_mask) {
  auto weight = weight_opt.value_or(Tensor());
  auto running_mean = running_mean_opt.value_or(Tensor());
  auto running_var = running_var_opt.value_or(Tensor());
  auto save_mean = save_mean_opt.value_or(Tensor());
  auto save_invstd = save_invstd_opt.value_or(Tensor());
  if (!hpu_check_inputs_impl(
          "native_batch_norm_backward",
          {grad_out,
           input,
           weight,
           running_mean,
           running_var,
           save_mean,
           save_invstd}))
    return AtenHpuTypeDefault::native_batch_norm_backward(
        grad_out,
        input,
        weight_opt,
        running_mean_opt,
        running_var_opt,
        save_mean_opt,
        save_invstd_opt,
        train,
        eps,
        output_mask);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return batch_norm_bwd_hpu_lazy(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask);

  } else {
    return batch_norm_bwd_hpu(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask);
  }
}

std::tuple<Tensor, Tensor, Tensor> hpu_wrap::native_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    double eps) {
  if (!hpu_check_inputs_impl(
          "native_layer_norm",
          {input, weight_opt.value_or(Tensor()), bias_opt.value_or(Tensor())}))
    return AtenHpuTypeDefault::native_layer_norm(
        input, normalized_shape, weight_opt, bias_opt, eps);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return layer_norm_hpu_lazy(
        input, normalized_shape, weight_opt, bias_opt, eps);

  } else {
    return layer_norm_hpu(input, normalized_shape, weight_opt, bias_opt, eps);
  }
};
std::tuple<Tensor, Tensor, Tensor> hpu_wrap::native_layer_norm_backward(
    const at::Tensor& dY,
    const at::Tensor& X,
    IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    std::array<bool, 3> grad_input_mask) {
  if (!hpu_check_inputs_impl(
          "native_layer_norm_backward",
          {dY, X, mean, rstd, weight_opt.value_or(Tensor())}))
    return AtenHpuTypeDefault::native_layer_norm_backward(
        dY,
        X,
        normalized_shape,
        mean,
        rstd,
        weight_opt,
        bias_opt,
        grad_input_mask);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return layer_norm_backward_hpu_lazy(
        dY,
        X,
        normalized_shape,
        mean,
        rstd,
        weight_opt,
        bias_opt,
        grad_input_mask);

  } else {
    return layer_norm_backward_hpu(
        dY,
        X,
        normalized_shape,
        mean,
        rstd,
        weight_opt,
        bias_opt,
        grad_input_mask);
  }
};

Tensor hpu_wrap::norm(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& p,
    at::IntArrayRef dim,
    bool keepdim) {
  if (!hpu_check_inputs_impl("norm", {self}))
    return AtenHpuTypeDefault::norm(self, p, dim, keepdim);

  // TODO Implement correct variant
  static_cast<void>(dim);
  static_cast<void>(keepdim);
  return hpu_wrap::norm(self, p.value_or(2));
}

Tensor hpu_wrap::norm(const Tensor& self, const c10::Scalar& p) {
  if (!hpu_check_inputs_impl("norm", {self}))
    return AtenHpuTypeDefault::norm(self, p);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return norm_scalar_hpu_lazy(self, p);

  } else {
    return norm_scalar_hpu(self, p);
  }
}

Tensor hpu_wrap::norm(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& p,
    at::IntArrayRef dim,
    bool keepdim,
    at::ScalarType dtype) {
  if (!hpu_check_inputs_impl("norm", {self}))
    return AtenHpuTypeDefault::norm(self, p, dim, keepdim, dtype);
  // when both dtype and grad are float avoid fallback
  if (dtype == c10::ScalarType::Float &&
      self.scalar_type() == c10::ScalarType::Float) {
    return hpu_wrap::norm(self, p, dim, keepdim);
  } else {
    return AtenHpuTypeDefault::norm(self, p, dim, keepdim, dtype);
  }
}

Tensor hpu_wrap::frobenius_norm(const Tensor& self) {
  if (!hpu_check_inputs_impl("frobenius_norm", {self}))
    return AtenHpuTypeDefault::frobenius_norm(self);

  struct FrobeniusNorm : public torch::autograd::Function<FrobeniusNorm> {
    static at::Tensor forward(
        torch::autograd::AutogradContext*,
        const at::Tensor& self) {
      return frobenius_norm_hpu_lazy(self);
    }

    // Implemented for convention, not to be invoked
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext*,
        const torch::autograd::variable_list&) {
      HABANA_ASSERT(
          0 &&
          "autograd::Function<FrobeniusNorm>::backward - should not be reached.");
      return {};
    }
  };

  return FrobeniusNorm::apply(self);
}

Tensor hpu_wrap::frobenius_norm(
    const Tensor& self,
    at::IntArrayRef dim,
    bool keepdim) {
  if (!hpu_check_inputs_impl("frobenius_norm", {self}))
    return AtenHpuTypeDefault::frobenius_norm(self);
  static_cast<void>(dim);
  static_cast<void>(keepdim);

  struct FrobeniusNorm : public torch::autograd::Function<FrobeniusNorm> {
    static at::Tensor forward(
        torch::autograd::AutogradContext*,
        const at::Tensor& self) {
      return frobenius_norm_hpu_lazy(self);
    }

    // Implemented for convention, not to be invoked
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext*,
        const torch::autograd::variable_list&) {
      HABANA_ASSERT(
          0 &&
          "autograd::Function<FrobeniusNorm>::backward - should not be reached.");
      return {};
    }
  };

  return FrobeniusNorm::apply(self);
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
  // Note: Legacy eager mode is not supported
  auto weight = weight_opt.value_or(Tensor());
  auto bias = bias_opt.value_or(Tensor());

  TORCH_CHECK(weight.defined(), "undefined weight is not supported");
  TORCH_CHECK(bias.defined(), "undefined bias is not supported");

  auto running_mean = running_mean_opt.value_or(Tensor());
  auto running_var = running_var_opt.value_or(Tensor());

  if (GET_ENV_FLAG(PT_HPU_DISABLE_INSTANCE_NORM)) {
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

std::tuple<Tensor, Tensor> hpu_wrap::max_pool2d_with_indices(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (!hpu_check_inputs_impl("max_pool2d_with_indices", {input}))
    return AtenHpuTypeDefault::max_pool2d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return max_pool2d_with_indices_hpu_lazy(
        input, kernel_size, stride, padding, dilation, ceil_mode);
  } else {
    return max_pool2d_with_indices_hpu(
        input, kernel_size, stride, padding, dilation, ceil_mode);
  }
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
  if (!hpu_check_inputs_impl(
          "max_pool2d_with_indices_backward_out",
          {grad_input, grad_output, input, indices}))
    return AtenHpuTypeDefault::max_pool2d_with_indices_backward_out(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices,
        grad_input);

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

Tensor hpu_wrap::max_pool2d_with_indices_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  if (!hpu_check_inputs_impl(
          "max_pool2d_with_indices_backward", {grad_output, input, indices}))
    return AtenHpuTypeDefault::max_pool2d_with_indices_backward(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return max_pool2d_with_indices_backward_hpu_lazy(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices);
  } else {
    return max_pool2d_with_indices_backward_hpu(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices);
  }
};
Tensor hpu_wrap::avg_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(input),
      IValue(kernel_size),
      IValue(stride),
      IValue(padding),
      IValue(ceil_mode),
      IValue(count_include_pad),
      IValue(divisor_override)};
  check_handle->hpu_check_ivalues("avg_pool2d", op_stack);
  if (!(hpu_check_inputs_impl("avg_pool2d", {input}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return avg_pool2d_hpu_lazy(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);

  } else {
    return avg_pool2d_hpu(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
  }
};
Tensor& hpu_wrap::avg_pool2d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& grad_input) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(grad_output),
      IValue(input),
      IValue(kernel_size),
      IValue(stride),
      IValue(padding),
      IValue(ceil_mode),
      IValue(count_include_pad),
      IValue(divisor_override),
      IValue(grad_input)};
  check_handle->hpu_check_ivalues("avg_pool2d_backward_out", op_stack);
  if (!(hpu_check_inputs_impl(
            "avg_pool2d_backward_out", {grad_input, grad_output, input}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::avg_pool2d_backward_out(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        grad_input);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return avg_pool2d_backward_out_hpu_lazy(
        grad_input,
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);

  } else {
    return avg_pool2d_backward_out_hpu(
        grad_input,
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
  }
};
Tensor hpu_wrap::avg_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(grad_output),
      IValue(input),
      IValue(kernel_size),
      IValue(stride),
      IValue(padding),
      IValue(ceil_mode),
      IValue(count_include_pad),
      IValue(divisor_override)};
  check_handle->hpu_check_ivalues("avg_pool2d_backward", op_stack);
  if (!(hpu_check_inputs_impl("avg_pool2d_backward", {grad_output, input}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::avg_pool2d_backward(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return avg_pool2d_backward_hpu_lazy(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);

  } else {
    return avg_pool2d_backward_hpu(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
  }
};
Tensor& hpu_wrap::uniform_(
    Tensor& self,
    double from,
    double to,
    c10::optional<Generator> gen) {
  if (!hpu_check_inputs_impl("uniform_", {self}))
    return AtenHpuTypeDefault::uniform_(self, from, to, gen);

  return uniform_hpu(self, from, to, gen);
}

Tensor& hpu_wrap::normal_(
    Tensor& self,
    double mean,
    double std,
    c10::optional<Generator> gen) {
  if (!hpu_check_inputs_impl("normal_", {self}))
    return AtenHpuTypeDefault::normal_(self, mean, std, gen);

  return normal_hpu(self, mean, std, gen);
}

Tensor hpu_wrap::bernoulli(const Tensor& self, c10::optional<Generator> gen) {
  if (!hpu_check_inputs_impl("bernoulli", {self}))
    return AtenHpuTypeDefault::bernoulli(self, gen);

  return bernoulli_hpu(self, gen);
}

Tensor& hpu_wrap::bernoulli_(
    Tensor& self,
    double p,
    c10::optional<Generator> gen) {
  if (!hpu_check_inputs_impl("bernoulli_", {self}))
    return AtenHpuTypeDefault::bernoulli_(self, p, gen);

  return bernoulli_scalar_hpu(self, p, gen);
}

Tensor& hpu_wrap::randperm_out(
    int64_t n,
    c10::optional<Generator> gen,
    Tensor& out) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return randperm_hpu_lazy(out, n, gen);
  } else {
    return randperm_hpu(out, n, gen);
  }
}

std::tuple<Tensor, Tensor> hpu_wrap::_fused_dropout(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen) {
  if (!hpu_check_inputs_impl("_fused_dropout", {self}))
    return AtenHpuTypeDefault::_fused_dropout(self, p, gen);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return fused_dropout_hpu_lazy(self, p, gen);
  } else {
    return fused_dropout_hpu(self, p, gen);
  }
}

at::Tensor hpu_wrap::repeat(const at::Tensor& self, at::IntArrayRef repeats) {
  if (!hpu_check_inputs_impl("repeat", {self}))
    return AtenHpuTypeDefault::repeat(self, repeats);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return repeat_hpu_lazy(self, repeats);
  } else {
    return repeat_hpu(self, repeats);
  }
}

Tensor hpu_wrap::sum(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  if (!hpu_check_inputs_impl("sum_dim_IntList", {self}))
    return AtenHpuTypeDefault::sum(self, dim, keepdim, dtype);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sum_dim_IntList_hpu_lazy(self, dim, keepdim, dtype);

  } else {
    return sum_dim_IntList_hpu(self, dim, keepdim, dtype);
  }
};
Tensor& hpu_wrap::sum_out(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype,
    Tensor& output) {
  if (!hpu_check_inputs_impl("sum_out", {output, self}))
    return AtenHpuTypeDefault::sum_out(self, dim, keepdim, dtype, output);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sum_out_hpu_lazy(self, dim, keepdim, dtype, output);
  } else {
    return sum_IntList_out_hpu(self, dim, keepdim, dtype, output);
  }
};

Tensor hpu_wrap::cumsum(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  if (!hpu_check_inputs_impl("cumsum", {self}))
    return AtenHpuTypeDefault::cumsum(self, dim, dtype);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return cumsum_hpu_lazy(self, dim, dtype);
  } else {
    return cumsum_hpu(self, dim, dtype);
  }
};

Tensor hpu_wrap::mean(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  if (!hpu_check_inputs_impl("mean", {self}))
    return AtenHpuTypeDefault::mean(self, dim, keepdim, dtype);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mean_dim_hpu_lazy(self, dim, keepdim, dtype);

  } else {
    return mean_dim_hpu(self, dim, keepdim, dtype);
  }
};
Tensor& hpu_wrap::mean_out(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype,
    Tensor& output) {
  if (!hpu_check_inputs_impl("mean_out", {output, self}))
    return AtenHpuTypeDefault::mean_out(self, dim, keepdim, dtype, output);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mean_dim_out_hpu_lazy(output, self, dim, keepdim, dtype);

  } else {
    return mean_dim_out_hpu(output, self, dim, keepdim, dtype);
  }
};
Tensor hpu_wrap::sum(const Tensor& self, c10::optional<ScalarType> dtype) {
  if (!hpu_check_inputs_impl("sum", {self}))
    return AtenHpuTypeDefault::sum(self, dtype);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sum_hpu_lazy(self, dtype);

  } else {
    return sum_hpu(self, dtype);
  }
};
Tensor hpu_wrap::mean(const Tensor& self, c10::optional<ScalarType> dtype) {
  if (!hpu_check_inputs_impl("mean", {self}))
    return AtenHpuTypeDefault::mean(self, dtype);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return mean_hpu_lazy(self, dtype);

  } else {
    return mean_hpu(self, dtype);
  }
};
Tensor hpu_wrap::prod(const Tensor& self, c10::optional<ScalarType> dtype) {
  if (!hpu_check_inputs_impl("prod", {self}))
    return AtenHpuTypeDefault::prod(self, dtype);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return prod_hpu_lazy(self, dtype);

  } else {
    return prod_hpu(self, dtype);
  }
};
Tensor hpu_wrap::prod(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  if (!hpu_check_inputs_impl("prod", {self}))
    return AtenHpuTypeDefault::prod(self, dim, keepdim, dtype);

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
  if (!hpu_check_inputs_impl("max", {self}))
    return AtenHpuTypeDefault::max(self, dim, keepdim);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return max_dim_hpu_lazy(self, dim, keepdim);
  } else {
    return max_dim_hpu(self, dim, keepdim);
  }
};
at::Tensor hpu_wrap::max(const at::Tensor& self) {
  if (!hpu_check_inputs_impl("max", {self}))
    return AtenHpuTypeDefault::max(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return max_hpu_lazy(self);
  } else {
    return max_hpu(self);
  }
};

at::Tensor hpu_wrap::min(const at::Tensor& self) {
  if (!hpu_check_inputs_impl("min", {self}))
    return AtenHpuTypeDefault::min(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return min_hpu_lazy(self);
  } else {
    return min_hpu(self);
  }
};

Tensor& hpu_wrap::any_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& output) {
  if (!hpu_check_inputs_impl("any", {self, output}))
    return AtenHpuTypeDefault::any_out(self, dim, keepdim, output);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return any_dim_out_hpu_lazy(self, dim, keepdim, output);
  } else {
    return any_dim_out_hpu(self, dim, keepdim, output);
  }
}
Tensor hpu_wrap::any(const Tensor& self, int64_t dim, bool keepdim) {
  if (!hpu_check_inputs_impl("any", {self}))
    return AtenHpuTypeDefault::any(self, dim, keepdim);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return any_dim_hpu_lazy(self, dim, keepdim);

  } else {
    return any_dim_hpu(self, dim, keepdim);
  }
};
Tensor hpu_wrap::any(const Tensor& self) {
  if (!hpu_check_inputs_impl("any", {self}))
    return AtenHpuTypeDefault::any(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return any_hpu_lazy(self);

  } else {
    return any_hpu(self);
  }
}

Tensor hpu_wrap::one_hot(const Tensor& self, int64_t num_classes) {
  if (!hpu_check_inputs_impl("one_hot", {self}))
    return AtenHpuTypeDefault::one_hot(self, num_classes);
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

Tensor hpu_wrap::_log_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(dim), IValue(half_to_float)};
  check_handle->hpu_check_ivalues("_log_softmax", op_stack);
  if (!(hpu_check_inputs_impl("_log_softmax", {self}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::_log_softmax(self, dim, half_to_float);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return log_softmax_hpu_lazy(self, dim, half_to_float);

  } else {
    return log_softmax_hpu(self, dim, half_to_float);
  }
}

Tensor hpu_wrap::_log_softmax_backward_data(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  if (!hpu_check_inputs_impl(
          "_log_softmax_backward_data", {grad, output, input}))
    return AtenHpuTypeDefault::_log_softmax_backward_data(
        grad, output, dim, input);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return log_softmax_backward_hpu_lazy(grad, output, dim, input);

  } else {
    return log_softmax_backward_hpu(grad, output, dim, input);
  }
};
Tensor hpu_wrap::_softmax(
    const Tensor& self,
    int64_t dim,
    const bool half_to_float) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(dim), IValue(half_to_float)};
  check_handle->hpu_check_ivalues("_softmax", op_stack);
  if (!(hpu_check_inputs_impl("_softmax", {self}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::_softmax(self, dim, half_to_float);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return softmax_hpu_lazy(self, dim, half_to_float);

  } else {
    return softmax_hpu(self, dim, half_to_float);
  }
};
Tensor hpu_wrap::_softmax_backward_data(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  if (!hpu_check_inputs_impl("_softmax_backward_data", {grad, output, input}))
    return AtenHpuTypeDefault::_softmax_backward_data(grad, output, dim, input);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return softmax_backward_hpu_lazy(grad, output, dim, input);

  } else {
    return softmax_backward_hpu(grad, output, dim, input);
  }
};
struct SoftmaxFunction : public torch::autograd::Function<SoftmaxFunction> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor input,
      int64_t dim,
      c10::optional<at::ScalarType> dtype) {
    at::Tensor result;
    if ((input.scalar_type() != c10::ScalarType::BFloat16) &&
        (input.scalar_type() != c10::ScalarType::Float)) {
      Tensor converted =
          dtype.has_value() ? input.toType(dtype.value()) : input;
      result = hpu_wrap::_softmax(converted, dim, false);
    } else {
      result = hpu_wrap::_softmax(input, dim, false);
    }
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
    auto result =
        hpu_wrap::_softmax_backward_data(grad_output[0], output, dim, input);
    return {result, torch::Tensor(), torch::Tensor()};
  }
};

Tensor hpu_wrap::softmax(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  return SoftmaxFunction::apply(self, dim, dtype);
}

Tensor hpu_wrap::empty(
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  at::TensorOptions options = at::TensorOptions()
                                  .dtype(std::move(dtype))
                                  .layout(std::move(layout))
                                  .pinned_memory(std::move(pin_memory))
                                  .device(std::move(device));

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
  if (!hpu_check_inputs_impl("empty_strided", {}))
    return AtenHpuTypeDefault::empty_strided(
        size, stride, dtype, layout, device, pin_memory);

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

Tensor hpu_wrap::clone(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format) {
  if (!hpu_check_inputs_impl("clone", {self}))
    return AtenHpuTypeDefault::clone(self, memory_format);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return clone_hpu_lazy(self, memory_format);

  } else {
    return clone_hpu(self, memory_format);
  }
};
Tensor& hpu_wrap::zero_(Tensor& self) {
  if (!hpu_check_inputs_impl("zero_", {self}))
    return AtenHpuTypeDefault::zero_(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return zero_hpu_lazy(self);

  } else {
    return zero_hpu(self);
  }
};
Tensor hpu_wrap::cat(const TensorList tensors, int64_t dim_) {
  if (!hpu_check_inputs_impl("cat", {tensors[0]}))
    return AtenHpuTypeDefault::cat(tensors, dim_);

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
  if (!hpu_check_inputs_impl("cat_out", {result, tensors[0]}))
    return AtenHpuTypeDefault::cat_out(tensors, dim_, result);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return cat_hpu_lazy_out(result, tensors, dim_);

  } else {
    return cat_hpu_out(result, tensors, dim_);
  }
};
Tensor hpu_wrap::transpose(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  if (!hpu_check_inputs_impl("transpose", {self}))
    return AtenHpuTypeDefault::transpose(self, dim0_, dim1_);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return transpose_hpu_lazy(self, dim0_, dim1_);

  } else {
    return transpose_hpu(self, dim0_, dim1_);
  }
};
Tensor& hpu_wrap::transpose_(Tensor& self, int64_t dim0_, int64_t dim1_) {
  if (!hpu_check_inputs_impl("transpose_", {self}))
    return AtenHpuTypeDefault::transpose_(self, dim0_, dim1_);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return transpose_hpu_lazy_(self, dim0_, dim1_);

  } else {
    return transpose_hpu_(self, dim0_, dim1_);
  }
};
Tensor hpu_wrap::t(const Tensor& self) {
  if (!hpu_check_inputs_impl("t", {self}))
    return AtenHpuTypeDefault::t(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return t_hpu_lazy(self);

  } else {
    return t_hpu(self);
  }
};
Tensor& hpu_wrap::t_(Tensor& self) {
  if (!hpu_check_inputs_impl("t_", {self}))
    return AtenHpuTypeDefault::t_(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return t_hpu_lazy_(self);

  } else {
    return t_hpu_(self);
  }
};
Tensor hpu_wrap::permute(const Tensor& self, IntArrayRef dims_) {
  if (!hpu_check_inputs_impl("permute", {self}))
    return AtenHpuTypeDefault::permute(self, dims_);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return permute_hpu_lazy(self, dims_);

  } else {
    return permute_hpu(self, dims_);
  }
};
Tensor hpu_wrap::expand(const Tensor& self, IntArrayRef size, bool implicit) {
  if (!hpu_check_inputs_impl("expand", {self}))
    return AtenHpuTypeDefault::expand(self, size, implicit);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return expand_hpu_lazy(self, size, implicit);

  } else {
    return expand_hpu(self, size, implicit);
  }
};
std::vector<Tensor> hpu_wrap::split_with_sizes(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  if (!hpu_check_inputs_impl("split_with_sizes", {self}))
    return AtenHpuTypeDefault::split_with_sizes(self, split_sizes, dim);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return split_with_sizes_hpu_lazy(self, split_sizes, dim);

  } else {
    return split_with_sizes_hpu(self, split_sizes, dim);
  }
};
Tensor hpu_wrap::threshold_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(grad_output), IValue(self), IValue(threshold)};
  check_handle->hpu_check_ivalues("threshold_backward", op_stack);
  if (!(hpu_check_inputs_impl("threshold_backward", {grad_output, self}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::threshold_backward(grad_output, self, threshold);
  return threshold_backward_hpu(grad_output, self, threshold);
};
std::tuple<Tensor&, Tensor&> hpu_wrap::topk_out(
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self),
      IValue(k),
      IValue(dim_),
      IValue(largest),
      IValue(sorted),
      IValue(values),
      IValue(indices)};
  check_handle->hpu_check_ivalues("topk", op_stack);
  if (!(hpu_check_inputs_impl("topk_out", {values, indices, self}) &&
        check_handle->get_status()))
    return AtenHpuTypeDefault::topk_out(
        self, k, dim_, largest, sorted, values, indices);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return topk_out_hpu_lazy(values, indices, self, k, dim_, largest, sorted);

  } else {
    return topk_out_hpu(values, indices, self, k, dim_, largest, sorted);
  }
};
std::tuple<Tensor, Tensor> hpu_wrap::topk(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(k), IValue(dim), IValue(largest), IValue(sorted)};
  check_handle->hpu_check_ivalues("topk", op_stack);
  if (!(hpu_check_inputs_impl("topk", {self}) && check_handle->get_status()))
    return AtenHpuTypeDefault::topk(self, k, dim, largest, sorted);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return topk_hpu_lazy(self, k, dim, largest, sorted);

  } else {
    return topk_hpu(self, k, dim, largest, sorted);
  }
};
std::tuple<Tensor, Tensor> hpu_wrap::sort(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(dim), IValue(descending)};
  check_handle->hpu_check_ivalues("sort", op_stack);
  if (!(hpu_check_inputs_impl("sort", {self}) && check_handle->get_status()))
    return AtenHpuTypeDefault::sort(self, dim, descending);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sort_hpu_lazy(self, dim, descending);

  } else {
    return sort_hpu(self, dim, descending);
  }
};

at::Tensor hpu_wrap::elu(
    const at::Tensor& self,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(alpha), IValue(scale), IValue(input_scale)};
  check_handle->hpu_check_ivalues("elu", op_stack);
  if (!(hpu_check_inputs_impl(__func__, {self}) && check_handle->get_status()))
    return AtenHpuTypeDefault::elu(self, alpha, scale, input_scale);
  return elu_hpu_lazy(self, alpha, scale, input_scale);
}

at::Tensor& hpu_wrap::elu_(
    at::Tensor& self,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale) {
  OpAttributeCheck* check_handle = OpAttributeCheck::get_instance();
  std::vector<c10::IValue> op_stack = {
      IValue(self), IValue(alpha), IValue(scale), IValue(input_scale)};
  check_handle->hpu_check_ivalues("elu", op_stack);
  if (!(hpu_check_inputs_impl(__func__, {self}) && check_handle->get_status()))
    return AtenHpuTypeDefault::elu_(self, alpha, scale, input_scale);
  return elu_hpu_lazy_(self, alpha, scale, input_scale);
}

Tensor hpu_wrap::relu(const Tensor& input) {
  if (!hpu_check_inputs_impl("relu", {input}))
    return AtenHpuTypeDefault::relu(input);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return relu_hpu_lazy(input);
  } else {
    return relu_hpu(input);
  }
};
Tensor& hpu_wrap::relu_(Tensor& self) {
  if (!hpu_check_inputs_impl("relu_", {self}))
    return AtenHpuTypeDefault::relu_(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return relu_hpu_lazy_(self);
  } else {
    return relu_hpu_(self);
  }
}

Tensor& hpu_wrap::leaky_relu_(Tensor& self, const Scalar& negative_slope) {
  if (!hpu_check_inputs_impl("leaky_relu_", {self}))
    return AtenHpuTypeDefault::leaky_relu_(self, negative_slope);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return leaky_relu_lazy_(self, negative_slope);
  } else {
    return leaky_relu_hpu_(self, negative_slope);
  }
}

at::Tensor hpu_wrap::leaky_relu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negative_slope,
    bool self_is_result) {
  if (!hpu_check_inputs_impl("leaky_relu_backward", {grad_output, self}))
    return AtenHpuTypeDefault::leaky_relu_backward(
        grad_output, self, negative_slope, self_is_result);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return leaky_relu_backward_lazy(
        grad_output, self, negative_slope, self_is_result);
  } else {
    return leaky_relu_backward_hpu(
        grad_output, self, negative_slope, self_is_result);
  }
}

at::Tensor hpu_wrap::leaky_relu(
    const at::Tensor& self,
    const at::Scalar& negative_slope) {
  if (!hpu_check_inputs_impl("leaky_relu", {self}))
    return AtenHpuTypeDefault::leaky_relu(self, negative_slope);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return leaky_relu_lazy(self, negative_slope);
  } else {
    return leaky_relu_hpu(self, negative_slope);
  }
}

Tensor hpu_wrap::sigmoid(const Tensor& input) {
  if (!hpu_check_inputs_impl("sigmoid", {input}))
    return AtenHpuTypeDefault::sigmoid(input);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sigmoid_hpu_lazy(input);

  } else {
    return sigmoid_hpu(input);
  }
};
Tensor hpu_wrap::sigmoid_backward(const Tensor& grad_in, const Tensor& input) {
  if (!hpu_check_inputs_impl("sigmoid_backward", {grad_in, input}))
    return AtenHpuTypeDefault::sigmoid_backward(grad_in, input);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sigmoid_backward_hpu_lazy(grad_in, input);

  } else {
    return sigmoid_backward_hpu(grad_in, input);
  }
};

at::Tensor& hpu_wrap::hardsigmoid_(at::Tensor& self) {
  if (!hpu_check_inputs_impl(__func__, {self}))
    return AtenHpuTypeDefault::hardsigmoid_(self);

  return hardsigmoid_hpu_lazy_(self);
}

at::Tensor hpu_wrap::hardsigmoid(const at::Tensor& self) {
  if (!hpu_check_inputs_impl(__func__, {self}))
    return AtenHpuTypeDefault::hardsigmoid(self);
  return hardsigmoid_hpu_lazy(self);
}

at::Tensor hpu_wrap::hardsigmoid_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  if (!hpu_check_inputs_impl(__func__, {grad_output, self}))
    return AtenHpuTypeDefault::hardsigmoid_backward(grad_output, self);
  return hardsigmoid_backward_hpu_lazy(grad_output, self);
}

Tensor hpu_wrap::sqrt(const Tensor& input) {
  if (!hpu_check_inputs_impl("sqrt", {input}))
    return AtenHpuTypeDefault::sqrt(input);
  return sqrt_hpu(input);
};

Tensor hpu_wrap::tanh(const Tensor& input) {
  if (!hpu_check_inputs_impl("tanh", {input}))
    return AtenHpuTypeDefault::tanh(input);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return tanh_hpu_lazy(input);

  } else {
    return tanh_hpu(input);
  }
};
Tensor& hpu_wrap::tanh_(Tensor& self) {
  if (!hpu_check_inputs_impl("tanh_", {self}))
    return AtenHpuTypeDefault::tanh_(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return tanh_hpu_lazy_(self);

  } else {
    return tanh_hpu_(self);
  }
};
Tensor& hpu_wrap::tanh_out(const Tensor& self, Tensor& out) {
  if (!hpu_check_inputs_impl("tanh_out", {out, self}))
    return AtenHpuTypeDefault::tanh_out(self, out);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return tanh_out_hpu_lazy(out, self);

  } else {
    return tanh_out_hpu(out, self);
  }
};
Tensor hpu_wrap::tanh_backward(const Tensor& grad_in, const Tensor& input) {
  if (!hpu_check_inputs_impl("tanh_backward", {grad_in, input}))
    return AtenHpuTypeDefault::tanh_backward(grad_in, input);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return tanh_backward_hpu_lazy(grad_in, input);

  } else {
    return tanh_backward_hpu(grad_in, input);
  }
};
Tensor hpu_wrap::gelu(const Tensor& self) {
  if (!hpu_check_inputs_impl("gelu", {self}))
    return AtenHpuTypeDefault::gelu(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return gelu_hpu_lazy(self);

  } else {
    return gelu_hpu(self);
  }
};
Tensor hpu_wrap::gelu_backward(const Tensor& grad, const Tensor& self) {
  if (!hpu_check_inputs_impl("gelu_backward", {grad, self}))
    return AtenHpuTypeDefault::gelu_backward(grad, self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return gelu_backward_hpu_lazy(grad, self);

  } else {
    return gelu_backward_hpu(grad, self);
  }
};

Tensor& hpu_wrap::erf_(Tensor& self) {
  if (!hpu_check_inputs_impl("erf_", {self}))
    return AtenHpuTypeDefault::erf_(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return erf_hpu_lazy_(self);

  } else {
    return erf_hpu_(self);
  }
};
Tensor hpu_wrap::erf(const Tensor& self) {
  if (!hpu_check_inputs_impl("erf", {self}))
    return AtenHpuTypeDefault::erf(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return erf_hpu_lazy(self);

  } else {
    return erf_hpu(self);
  }
};
Tensor& hpu_wrap::exp_(Tensor& self) {
  if (!hpu_check_inputs_impl("exp_", {self}))
    return AtenHpuTypeDefault::exp_(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return exp_hpu_lazy_(self);

  } else {
    return exp_hpu_(self);
  }
};
Tensor hpu_wrap::exp(const Tensor& self) {
  if (!hpu_check_inputs_impl("exp", {self}))
    return AtenHpuTypeDefault::exp(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return exp_hpu_lazy(self);

  } else {
    return exp_hpu(self);
  }
};
Tensor hpu_wrap::sign(const Tensor& self) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sign_hpu_lazy(self);

  } else {
    return sign_hpu(self);
  }
}
Tensor& hpu_wrap::sign_(at::Tensor& self) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sign_hpu_lazy_(self);

  } else {
    return sign_hpu_(self);
  }
}
Tensor hpu_wrap::sgn(const Tensor& self) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sgn_hpu_lazy(self);

  } else {
    return sgn_hpu(self);
  }
}
Tensor& hpu_wrap::sgn_(at::Tensor& self) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sgn_hpu_lazy_(self);

  } else {
    return sgn_hpu_(self);
  }
}
Tensor& hpu_wrap::neg_out(const Tensor& input, Tensor& result) {
  if (!hpu_check_inputs_impl("neg_out", {result, input}))
    return AtenHpuTypeDefault::neg_out(input, result);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return neg_out_hpu_lazy(result, input);

  } else {
    return neg_out_hpu(result, input);
  }
};
Tensor& hpu_wrap::reciprocal_(Tensor& self) {
  if (!hpu_check_inputs_impl("reciprocal_", {self}))
    return AtenHpuTypeDefault::reciprocal_(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return reciprocal_hpu_lazy_(self);

  } else {
    return reciprocal_hpu_(self);
  }
};
Tensor hpu_wrap::reciprocal(const Tensor& self) {
  if (!hpu_check_inputs_impl("reciprocal", {self}))
    return AtenHpuTypeDefault::reciprocal(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return reciprocal_hpu_lazy(self);

  } else {
    return reciprocal_hpu(self);
  }
};
Tensor& hpu_wrap::reciprocal_out(const Tensor& self, Tensor& result) {
  if (!hpu_check_inputs_impl("reciprocal_out", {result, self}))
    return AtenHpuTypeDefault::reciprocal_out(self, result);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return reciprocal_out_hpu_lazy(result, self);

  } else {
    return reciprocal_out_hpu(result, self);
  }
};
Tensor hpu_wrap::clamp_min(const Tensor& self, const Scalar& min) {
  if (!hpu_check_inputs_impl("clamp_min", {self}))
    return AtenHpuTypeDefault::clamp_min(self, min);
  return clamp_min_hpu(self, min);
};
Tensor& hpu_wrap::clamp_(
    Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  if (!hpu_check_inputs_impl("clamp_", {self}))
    return AtenHpuTypeDefault::clamp_(self, min, max);
  return clamp_hpu_(self, min, max);
};
Tensor hpu_wrap::clamp(
    const Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  if (!hpu_check_inputs_impl("clamp", {self}))
    return AtenHpuTypeDefault::clamp(self, min, max);
  return clamp_hpu(self, min, max);
};

Tensor hpu_wrap::isnan(const Tensor& self) {
  if (!hpu_check_inputs_impl("isnan", {self}))
    return AtenHpuTypeDefault::isnan(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return isnan_hpu_lazy(self);
  } else {
    return isnan_hpu(self);
  }
};

Tensor hpu_wrap::silu(const Tensor& self) {
  if (!hpu_check_inputs_impl("silu", {self}))
    return AtenHpuTypeDefault::silu(self);

  return silu_hpu(self);
};

Tensor hpu_wrap::silu_backward(const Tensor& grad, const Tensor& self) {
  if (!hpu_check_inputs_impl("silu_backward", {grad, self}))
    return AtenHpuTypeDefault::silu_backward(grad, self);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return silu_backward_hpu_lazy(grad, self);
  } else {
    HABANA_ASSERT(0 && "silu_backward not implemented for eager mode");
    return silu_backward_hpu_lazy(grad, self);
  }
};

Tensor hpu_wrap::abs(const Tensor& self) {
  if (!hpu_check_inputs_impl("abs", {self}))
    return AtenHpuTypeDefault::abs(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return abs_hpu_lazy(self);

  } else {
    return abs_hpu(self);
  }
};
Tensor& hpu_wrap::abs_(Tensor& self) {
  if (!hpu_check_inputs_impl("abs_", {self}))
    return AtenHpuTypeDefault::abs_(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    auto& t = abs_hpu_lazy_(self);
    return t;
  } else {
    return abs_hpu_(self);
  }
};
Tensor hpu_wrap::round(const Tensor& self) {
  if (!hpu_check_inputs_impl("round", {self}))
    return AtenHpuTypeDefault::round(self);
  return round_hpu(self);
};
Tensor& hpu_wrap::round_(Tensor& self) {
  if (!hpu_check_inputs_impl("round_", {self}))
    return AtenHpuTypeDefault::round_(self);
  return round_hpu_(self);
};
Tensor hpu_wrap::rsqrt(const Tensor& self) {
  if (!hpu_check_inputs_impl("rsqrt", {self}))
    return AtenHpuTypeDefault::rsqrt(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return rsqrt_hpu_lazy(self);

  } else {
    return rsqrt_hpu(self);
  }
};
Tensor& hpu_wrap::rsqrt_(Tensor& self) {
  if (!hpu_check_inputs_impl("rsqrt_", {self}))
    return AtenHpuTypeDefault::rsqrt_(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return rsqrt_hpu_lazy_(self);
  } else {
    return rsqrt_hpu_(self);
  }
};
Tensor hpu_wrap::neg(const Tensor& self) {
  if (!hpu_check_inputs_impl("neg", {self}))
    return AtenHpuTypeDefault::neg(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return neg_hpu_lazy(self);
  } else {
    return neg_hpu(self);
  }
};
Tensor hpu_wrap::sin(const Tensor& self) {
  if (!hpu_check_inputs_impl("sin", {self}))
    return AtenHpuTypeDefault::sin(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return sin_hpu_lazy(self);
  } else {
    return sin_hpu(self);
  }
};
Tensor hpu_wrap::cos(const Tensor& self) {
  if (!hpu_check_inputs_impl("cos", {self}))
    return AtenHpuTypeDefault::cos(self);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return cos_hpu_lazy(self);
  } else {
    return cos_hpu(self);
  }
};
Tensor hpu_wrap::floor(const Tensor& input) {
  if (!hpu_check_inputs_impl("floor", {input}))
    return AtenHpuTypeDefault::floor(input);
  return floor_hpu(input);
};
Tensor& hpu_wrap::floor_(Tensor& self) {
  if (!hpu_check_inputs_impl("floor_", {self}))
    return AtenHpuTypeDefault::floor_(self);
  return floor_hpu_(self);
};

Tensor hpu_wrap::log(const Tensor& input) {
  if (!hpu_check_inputs_impl("log", {input}))
    return AtenHpuTypeDefault::log(input);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return log_hpu_lazy(input);
  } else {
    return log_hpu(input);
  }
};
Tensor& hpu_wrap::log_(Tensor& self) {
  if (!hpu_check_inputs_impl("log_", {self}))
    return AtenHpuTypeDefault::log_(self);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return log_hpu_lazy_(self);
  } else {
    return log_hpu_(self);
  }
};
Tensor hpu_wrap::log2(const Tensor& input) {
  if (!hpu_check_inputs_impl("log2", {input}))
    return AtenHpuTypeDefault::log2(input);
  return log2_hpu(input);
};
Tensor& hpu_wrap::log2_(Tensor& self) {
  if (!hpu_check_inputs_impl("log2_", {self}))
    return AtenHpuTypeDefault::log2_(self);
  return log2_hpu_(self);
}
Tensor hpu_wrap::argmax(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return argmax_hpu_lazy(self, dim, keepdim);
  } else {
    return argmax_hpu(self, dim, keepdim);
  }
}

std::tuple<Tensor, Tensor, Tensor> hpu_wrap::_unique2(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return unique2_hpu_lazy(self, sorted, return_inverse, return_counts);
  } else {
    return unique2_hpu(self, sorted, return_inverse, return_counts);
  }
}

std::vector<at::Tensor> hpu_wrap::unbind(const at::Tensor& self, int64_t dim) {
  if (!hpu_check_inputs_impl("unbind", {self}))
    return AtenHpuTypeDefault::unbind(self, dim);

  return at::native::unbind(self, dim);
}

Tensor hpu_wrap::stack(TensorList tensors, int64_t dim) {
  if (!hpu_check_inputs_impl("stack", {tensors[0]}))
    return AtenHpuTypeDefault::stack(tensors, dim);

  return at::native::stack(tensors, dim);
}

Tensor hpu_wrap::alias(const at::Tensor& self) {
  return hpu_wrap::as_strided(
      self, self.sizes(), self.strides(), self.storage_offset());
}

Tensor hpu_wrap::_unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
  if (!hpu_check_inputs_impl("_unsafe_view", {self}))
    return AtenHpuTypeDefault::_unsafe_view(self, size);

  return at::native::_unsafe_view(self, size);
}

at::Tensor hpu_wrap::squeeze(const at::Tensor& self) {
  if (!hpu_check_inputs_impl("squeeze", {self}))
    return AtenHpuTypeDefault::squeeze(self);

  return at::native::squeeze(self);
}

at::Tensor hpu_wrap::squeeze(const at::Tensor& self, int64_t dim) {
  if (!hpu_check_inputs_impl("squeeze", {self}))
    return AtenHpuTypeDefault::squeeze(self, dim);

  return at::native::squeeze(self, dim);
}

at::Tensor& hpu_wrap::squeeze_(at::Tensor& self) {
  if (!hpu_check_inputs_impl("squeeze_", {self}))
    return AtenHpuTypeDefault::squeeze_(self);

  return at::native::squeeze_(self);
}

at::Tensor& hpu_wrap::squeeze_(at::Tensor& self, int64_t dim) {
  if (!hpu_check_inputs_impl("squeeze_", {self}))
    return AtenHpuTypeDefault::squeeze_(self, dim);

  return at::native::squeeze_(self, dim);
}

at::Tensor hpu_wrap::unsqueeze(const at::Tensor& self, int64_t dim) {
  if (!hpu_check_inputs_impl("unsqueeze", {self}))
    return AtenHpuTypeDefault::unsqueeze(self, dim);

  return at::native::unsqueeze(self, dim);
}

at::Tensor& hpu_wrap::unsqueeze_(at::Tensor& self, int64_t dim) {
  if (!hpu_check_inputs_impl("unsqueeze_", {self}))
    return AtenHpuTypeDefault::unsqueeze_(self, dim);

  return at::native::unsqueeze_(self, dim);
}

const at::Tensor& hpu_wrap::as_strided_(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  // No CPU fallback for as_strided_ since H2D & D2H DMA support only contiguous
  // tensor transfers
  // if (!hpu_check_inputs_impl("as_strided_", {self}))
  //  return AtenHpuTypeDefault::as_strided_(self, size, stride,
  //  storage_offset);

  return as_strided_hpu_lazy_(self, size, stride, storage_offset);
}

std::vector<at::Tensor> hpu_wrap::split(
    const at::Tensor& self,
    int64_t split_size,
    int64_t dim) {
  if (!hpu_check_inputs_impl("split", {self}))
    return AtenHpuTypeDefault::split(self, split_size, dim);

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
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  if (!hpu_check_inputs_impl("upsample_nearest2d", {input}))
    return AtenHpuTypeDefault::upsample_nearest2d(
        input, output_size, scale_factors);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return upsample_nearest2d_hpu_lazy(input, output_size, scale_factors);
  } else {
    return upsample_nearest2d_hpu(input, output_size, scale_factors);
  }
};

Tensor hpu_wrap::upsample_nearest2d_backward(
    const Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  if (!hpu_check_inputs_impl("upsample_nearest2d_backward", {grad_output}))
    return AtenHpuTypeDefault::upsample_nearest2d_backward(
        grad_output, output_size, input_size, scale_factors);

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
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  if (!hpu_check_inputs_impl("upsample_nearest3d", {input}))
    return AtenHpuTypeDefault::upsample_nearest3d(
        input, output_size, scale_factors);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return upsample_nearest3d_hpu_lazy(input, output_size, scale_factors);
  } else {
    return upsample_nearest3d_hpu(input, output_size, scale_factors);
  }
};

Tensor hpu_wrap::upsample_nearest3d_backward(
    const Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  if (!hpu_check_inputs_impl("upsample_nearest3d_backward", {grad_output}))
    return AtenHpuTypeDefault::upsample_nearest3d_backward(
        grad_output, output_size, input_size, scale_factors);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return upsample_nearest3d_backward_hpu_lazy(
        grad_output, output_size, input_size, scale_factors);
  } else {
    return upsample_nearest3d_backward_hpu(
        grad_output, output_size, input_size, scale_factors);
  }
};

Scalar hpu_wrap::_local_scalar_dense(const Tensor& self) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return _local_scalar_dense_hpu_lazy(self);
  } else {
    return _local_scalar_dense_hpu(self);
  }
}

Tensor& hpu_wrap::bitwise_and_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  if (!hpu_check_inputs_impl("bitwise_and_out", {out, self, other}))
    return AtenHpuTypeDefault::bitwise_and_out(self, other, out);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return bitwise_and_out_hpu_lazy(out, self, other);
  } else {
    return bitwise_and_out_hpu(out, self, other);
  }
}

/*
Tensor& hpu_wrap::bitwise_and_out(
    Tensor& out,
    const Tensor& self,
    Scalar other) {
  if (!hpu_check_inputs_impl("bitwise_and_out", {out, self});

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return bitwise_and_out_hpu_lazy(out, self, other);
  } else {
    return bitwise_and_out_hpu(out, self, other);
  }
}
*/

Tensor& hpu_wrap::bitwise_or_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  if (!hpu_check_inputs_impl("bitwise_or_out", {out, self, other}))
    return AtenHpuTypeDefault::bitwise_or_out(self, other, out);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return bitwise_or_out_hpu_lazy(out, self, other);
  } else {
    return bitwise_or_out_hpu(out, self, other);
  }
}

Tensor& hpu_wrap::bitwise_xor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  if (!hpu_check_inputs_impl("bitwise_xor_out", {out, self, other}))
    return AtenHpuTypeDefault::bitwise_xor_out(self, other, out);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return bitwise_xor_out_hpu_lazy(out, self, other);
  } else {
    return bitwise_xor_out_hpu(out, self, other);
  }
}

Tensor& hpu_wrap::bitwise_not_out(const Tensor& self, Tensor& out) {
  if (!hpu_check_inputs_impl("bitwise_not_out", {out, self}))
    return AtenHpuTypeDefault::bitwise_not_out(self, out);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return bitwise_not_out_hpu_lazy(out, self);
  } else {
    return bitwise_not_out_hpu(out, self);
  }
}

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
    at::Tensor& lr_t,
    at::Tensor& neg_step_t,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay) {
  TORCH_CHECK((weight_vec.size() > 0), "Can not process empty weight vector");
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
  TORCH_CHECK((grad.size() > 0), "Can not process empty grad vector");
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return fused_norm_hpu_lazy(grad, max_norm, norm_type);
  } else {
    return fused_norm_hpu(grad, max_norm, norm_type);
  }
}
Tensor& optimizer_adagrad_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_adagrad_hpu_lazy(
        gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
  } else {
    optimizer_adagrad_hpu(
        gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
  }

  return lr;
}
Tensor hpu_wrap::ones_like(
    const Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
  if (!hpu_check_inputs_impl("ones_like", {self}))
    return AtenHpuTypeDefault::ones_like(
        self, dtype, layout, device, pin_memory, memory_format);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return ones_like_hpu_lazy(
        self, dtype, layout, device, pin_memory, memory_format);
  } else {
    at::TensorOptions options = at::TensorOptions()
                                    .dtype(dtype)
                                    .layout(layout)
                                    .pinned_memory(pin_memory)
                                    .device(device);
    return ones_like_hpu(self, options, memory_format);
  }
}

Tensor& optimizer_sgd_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  if (!habana_lazy::isDeviceInLoweringMode(weights[0].device().index()) &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_sgd_hpu_lazy(gradients, weights, lr, wd, mom, damp, nesterov);
  } else {
    optimizer_sgd_hpu(gradients, weights, lr, wd, mom, damp, nesterov);
  }

  return lr;
}

Tensor& optimizer_sgd_momentum_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  if (!habana_lazy::isDeviceInLoweringMode(weights[0].device().index()) &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_sgd_momentum_hpu_lazy(
        gradients, weights, momentum, epoch_num, lr, wd, mom, damp, nesterov);
  } else {
    optimizer_sgd_momentum_hpu(
        gradients, weights, momentum, epoch_num, lr, wd, mom, damp, nesterov);
  }

  return lr;
}

Tensor optimizer_lamb_fused_norm_hpu_wrap(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm) {
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
    const std::vector<at::Tensor>& trust_ratio_vec,
    const float step,
    const float weight_decay,
    const int use_lamb) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_lamb_phase2_hpu_lazy(
        weight_vec,
        adam_norm_vec,
        weight_norm_vec,
        adam_step_vec,
        trust_ratio_vec,
        step,
        weight_decay,
        use_lamb);
  } else {
    optimizer_lamb_phase2_hpu(
        weight_vec,
        adam_norm_vec,
        weight_norm_vec,
        adam_step_vec,
        trust_ratio_vec,
        step,
        weight_decay,
        use_lamb);
  }
}

Tensor habana_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    float iou_threshold,
    float score_threshold) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return habana_nms_hpu_lazy(boxes, scores, iou_threshold, score_threshold);
  } else {
    return habana_nms_hpu(boxes, scores, iou_threshold, score_threshold);
  }
}

Tensor hpu_wrap::_masked_scale(
    const Tensor& self,
    const Tensor& mask,
    double scale) {
  if (!hpu_check_inputs_impl("_masked_scale", {self, mask}))
    return AtenHpuTypeDefault::_masked_scale(self, mask, scale);
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
  return IsfiniteFunction::apply(self);
}

struct MatmulFunction : public torch::autograd::Function<MatmulFunction> {
  static at::Tensor forward(
      AutogradContext* ctx,
      at::Tensor self,
      at::Tensor other) {
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
  return MatmulFunction::apply(self, other);
};

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
  return AdaptiveAvgPool2DFunction::apply(input, output_size);
};
struct SliceFunction : public torch::autograd::Function<SliceFunction> {
  static at::Tensor forward(
      AutogradContext* ctx,
      at::Tensor self,
      int64_t dim,
      c10::optional<int64_t> start,
      c10::optional<int64_t> end,
      int64_t step) {
    at::Tensor result;
    ctx->save_for_backward({self});
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["start"] = start;
    ctx->saved_data["end"] = end;
    ctx->saved_data["step"] = step;
    result = slice_hpu_lazy(self, dim, start, end, step);
    return result;
  }
  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    at::Tensor result;
    variable_list saved_vars = ctx->get_saved_variables();
    result = slice_backward_hpu_lazy(
        saved_vars[0],
        grad_output[0],
        ctx->saved_data["dim"].toInt(),
        ctx->saved_data["start"].toInt(),
        ctx->saved_data["end"].toInt(),
        ctx->saved_data["step"].toInt());
    return {
        result,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor()};
  }
};

Tensor hpu_wrap::slice(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return SliceFunction::apply(self, dim, start, end, step);
  } else {
    return slice_hpu(self, dim, start, end, step);
  }
};

Tensor& hpu_wrap::linspace_out(
    const Scalar& start,
    const Scalar& end,
    c10::optional<int64_t> steps,
    Tensor& out) {
  if (!steps.has_value()) {
    TORCH_WARN_ONCE(
        "Not providing a value for linspace's steps is deprecated and will "
        "throw a runtime error in a future release. This warning will appear "
        "only once per process.");
  }

  if (!hpu_check_inputs_impl("linspace_out", {out}))
    return AtenHpuTypeDefault::linspace_out(start, end, steps, out);

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return linspace_out_hpu_lazy(start, end, steps, out);
  } else {
    return linspace_out_hpu(start, end, steps, out);
  }
}

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
  return DropoutFunction::apply(input, p, train);
}

Tensor hpu_wrap::flip(const Tensor& self, IntArrayRef dims) {
  if (!hpu_check_inputs_impl("flip", {self}))
    return AtenHpuTypeDefault::flip(self, dims);

  // Lazy mode is not implemented
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return flip_hpu_lazy(self, dims);
  } else {
    return flip_hpu(self, dims);
  }
}

Tensor hpu_wrap::diag(const Tensor& self, int64_t diagonal) {
  if (!hpu_check_inputs_impl("diag", {self}))
    return AtenHpuTypeDefault::diag(self, diagonal);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return diag_hpu_lazy(self, diagonal);
  } else {
    return diag_hpu(self, diagonal);
  }
}

// Registration for all non-custom/aten ops are auto-generated and can be
// found in habana_kernels/aten_hpu_type_default.cpp.

TORCH_LIBRARY(hpu, m) {
  // Ops that need custom schema because of Generator
  m.def("geometric_(Tensor(a!) self, float p, int seed) -> Tensor(a!)");
  m.def(
      "uniform_(Tensor(a!) self, float from=0, float to=1, *, Tensor seed) -> Tensor(a!)");
  m.def(
      "normal_(Tensor(a!) self, float mean=0, float std=1, *, Tensor seed) -> Tensor(a!)");
  m.def("bernoulli(Tensor self, *, int seed) -> Tensor");
  m.def(
      "bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, int seed) -> Tensor(a!)");
  m.def(
      "bernoulli_.float(Tensor(a!) self, float p=0.5, *, int seed) -> Tensor(a!)");
  m.def(
      "multinomial(Tensor self, int num_samples, bool replacement=False, int seed=0) -> Tensor");
  m.def("random_(Tensor(a!) self, Tensor seed) -> Tensor(a!)");
  m.def(
      "random_.from(Tensor(a!) self, int from, int? to, Tensor seed) -> Tensor(a!)");
  m.def("random_.to(Tensor(a!) self, int to, Tensor seed) -> Tensor(a!)");

  m.def("nonzero(Tensor self) -> (Tensor Tensor)");
  m.def("mul_out(Tensor out, Tensor self, Tensor other) -> Tensor");
  m.def("div_out(Tensor out, Tensor self, Tensor other) -> Tensor");
  m.def(
      "bitwise_and_Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "bitwise_or_Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "bitwise_xor_Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("bitwise_not_Tensor_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("mm_t(Tensor mm, Tensor t , bool tr, bool no_tr) -> Tensor");
  m.def("habana_d2d_memcpy_other(Tensor s, Tensor d) -> Tensor");
  m.def(
      "sum_dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def(
      "prod_dim_Int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("all_dim(Tensor self, int dim, bool keepdim=False) -> Tensor");
  m.def(
      "scatter_value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor(a!)");
  m.def(
      "arange_out(Scalar start, Scalar end, Scalar step, Tensor result) -> Tensor(a!)");
  m.def("arange_out_ds(Tensor shape, Tensor result) -> Tensor(a!)");
  m.def("diag_out(Tensor self, int diagonal, Tensor output) -> Tensor");
  m.def(
      "randperm_out(int n, Generator? generator, Tensor output) -> Tensor(a!)");
  m.def(
      "max_dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
  m.def("habana_d2d_memcpy(Tensor self) -> (Tensor)");
  m.def(
      "habanaOptimizerSparseSgd(Tensor gradients, Tensor weights_in, Tensor moments_in, Tensor indices, Tensor learning_rate, Tensor valid_count_tensor, float mom, bool nesterov) -> (Tensor, Tensor)");
  m.def(
      "habanaOptimizerSparseAdagrad(Tensor gradients, Tensor weights_in, Tensor moments_in, Tensor indices, Tensor learning_rate, Tensor valid_count_tensor) -> (Tensor, Tensor)");
  m.def("cast(Tensor self, Scalar type) -> Tensor(a)");
  m.def(
      "embedding_bag_sum(Tensor input, Tensor indices, Tensor offsets, Tensor valid_count, int kernel_mode) -> (Tensor)");
  m.def(
      "embedding_bag_sum_bwd_out(Tensor out, Tensor input, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, int kernel_mode) -> (Tensor)");
  m.def(
      "habanaOptimizerFusedAdagrad(Tensor[] gradients, Tensor[] weights_in, Tensor[] variances_in, Tensor epoch_num, Tensor learning_rate, float wd, float lrd, float eps) -> Tensor(a!)");
  m.def(
      "habanaOptimizerFusedSGD(Tensor[] gradients, Tensor[] weights_in, Tensor learning_rate, float wd, float mom, float damp, bool nesterov) -> Tensor(a!)");
  m.def(
      "habanaOptimizerFusedSGDMomentum(Tensor[] gradients, Tensor[] weights_in, Tensor[] momentum_in, Tensor epoch_num, Tensor learning_rate, float wd, float mom, float damp, bool nesterov) -> Tensor(a!)");
  m.def(
      "hpu::habanaOptimizerAdamW(Tensor[] gradient_vec, Tensor[] weight_vec, Tensor[] exp_avg_vec, Tensor[] exp_avg_sq_vec, Tensor lr_t, Tensor neg_step_t, float beta1, float beta2, float epsilon, float weight_decay) -> (Tensor[])");
  m.def(
      "fused_norm_(Tensor[] grad, Tensor max_norm, float norm_type) -> (Tensor[])");
  m.def(
      "habanaOptimizerLambFusedNorm(Tensor[] grad, float max_norm, Tensor clip_norm) -> (Tensor)");
  m.def(
      "habanaOptimizerLambPhase1(Tensor[] grad, Tensor[] weights, Tensor[] exp_avg, Tensor[] exp_avg_sq, Tensor clip_global_grad_norm, float beta1, float beta2, float beta2, float epsilon, Tensor bias_corection1, Tensor bias_correction2, float weight_decay) -> (Tensor[], Tensor[], Tensor[])");
  m.def(
      "habanaOptimizerLambPhase2(Tensor[] weights, Tensor[] adam_norm, Tensor[] wt_norm, Tensor[] adam_step, Tensor[] trust_ratio, Tensor neg_step, float wd, int use_lamb) -> ()");
  m.def(
      "habana_nms(Tensor boxes, Tensor scores, float iou_threshold, float score_threshold) -> (Tensor, Tensor, Tensor)");
  m.def(
      "_unique2(Tensor self, bool sorted, bool return_inverse, bool return_counts) -> (Tensor, Tensor)");
  m.def(
      "gather_elements(Tensor self, Tensor index, Tensor? opt, int64_t dim_, bool sorted) -> Tensor");
  m.def("permute_cl(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("restride_cl(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("restride(Tensor(a) self, int[] dims) -> Tensor(a)");
  m.def("permute_weight(Tensor self, int[] size) -> (Tensor)");
  m.def("permuted_weight_restride(Tensor self, int[] size) -> (Tensor)");
  m.def("control_edge_other_(Tensor self, Tensor other) -> Tensor(a!)");
  m.def("control_edge_(Tensor self)-> Tensor(a!)");
  m.def(
      "hpu::native_batch_norm_rmv(Tensor input, Tensor? weight, Tensor? bias, Tensor? residual_add, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::native_batch_norm_inf(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor)");
  m.def(
      "as_strided_lazy_(Tensor self, int[] size, int[] stride, int offset, bool can_replace) -> (Tensor)");
  m.def(
      "as_strided_lazy_cl_(Tensor self, int[] size, int[] stride, int offset, bool can_replace) -> (Tensor)");
  m.def("as_strided_layout_(Tensor self, int[] size) -> (Tensor)");
  m.def("reshape(Tensor self, int[] size) -> (Tensor)");
  m.def(
      "matmul_backward(Tensor grad_out, Tensor self, Tensor other) -> (Tensor, Tensor)");
  m.def(
      "instance_norm(Tensor input, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "instance_norm_backward(Tensor input, Tensor grad_in, Tensor? mean, Tensor? istd, Tensor gamma) -> (Tensor, Tensor, Tensor)");
  m.def("view(Tensor input, Tensor shape) -> (Tensor)");
  m.def(
      "hpu::expand(Tensor(a) self, Tensor shape, *, bool implicit=False) -> Tensor(a)");
  m.def("hpu::repeat(Tensor self, Tensor repeats_shape) -> Tensor");
  m.def(
      "upsample_nearest2d_backward(Tensor grad_output, int[]? output_size, Tensor input_size, float[]? scale_factors) -> Tensor");
  m.def(
      "hpu::topk(Tensor self, Tensor k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)");
  m.def(
      "hpu::scatter_nd_onnx(Tensor input, Tensor indices, Tensor values) -> (Tensor)");
  m.def(
      "hpu::scatter_nd(Tensor input, Tensor indices, Tensor grouped_indices, Tensor update_locations, Tensor updates) -> (Tensor)");
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("habana_d2d_memcpy", habana_d2d_memcpy);
  m.impl("embedding_bag_sum", embedding_bag_sum_hpu_wrap);
  m.impl(
      "embedding_bag_sum_bwd_out",
      embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap);
  m.impl(
      "sum_dim_IntList",
      static_cast<at::Tensor (*)(
          const at::Tensor&,
          at::IntArrayRef,
          bool,
          c10::optional<at::ScalarType>)>(&hpu_wrap::sum));
  m.impl(
      "prod_dim_Int",
      static_cast<at::Tensor (*)(
          const at::Tensor&, int64_t, bool, c10::optional<at::ScalarType>)>(
          &hpu_wrap::prod));
  m.impl(
      "scatter_value",
      static_cast<
          at::Tensor& (*)(at::Tensor&, int64_t, const at::Tensor&, const Scalar&)>(
          &hpu_wrap::scatter_));
  m.impl(
      "arange_out",
      static_cast<
          at::Tensor& (*)(const Scalar&, const Scalar&, const Scalar&, at::Tensor&)>(
          &hpu_wrap::arange_out));
  m.impl(
      "randperm_out",
      static_cast<
          at::Tensor& (*)(int64_t, c10::optional<at::Generator>, at::Tensor&)>(
          &hpu_wrap::randperm_out));
  m.impl(
      "bernoulli_float",
      static_cast<
          at::Tensor& (*)(at::Tensor&, double p, c10::optional<at::Generator>)>(
          &hpu_wrap::bernoulli_));
  m.impl(
      "bitwise_and_Tensor_out",
      static_cast<
          at::Tensor& (*)(const at::Tensor&, const at::Tensor&, at::Tensor&)>(
          &hpu_wrap::bitwise_and_out));
  m.impl(
      "bitwise_or_Tensor_out",
      static_cast<
          at::Tensor& (*)(const at::Tensor&, const at::Tensor&, at::Tensor&)>(
          &hpu_wrap::bitwise_or_out));
  m.impl(
      "bitwise_xor_Tensor_out",
      static_cast<
          at::Tensor& (*)(const at::Tensor&, const at::Tensor&, at::Tensor&)>(
          &hpu_wrap::bitwise_xor_out));
  m.impl(
      "bitwise_not_Tensor_out",
      static_cast<at::Tensor& (*)(const at::Tensor&, at::Tensor&)>(
          &hpu_wrap::bitwise_not_out));
  m.impl(
      "max_dim",
      static_cast<std::tuple<at::Tensor, at::Tensor> (*)(
          const at::Tensor&, int64_t, bool)>(&hpu_wrap::max));
  m.impl(
      "matmul_backward",
      static_cast<std::tuple<at::Tensor, at::Tensor> (*)(
          const at::Tensor&, const at::Tensor&, const at::Tensor&)>(
          &matmul_backward_hpu));
}
