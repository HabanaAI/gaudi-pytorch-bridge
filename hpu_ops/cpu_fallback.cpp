/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "cpu_fallback.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/lazy_executor.h"

namespace habana {

bool isInplaceOp(std::string op_name) {
  auto pos = op_name.find('.');
  if (pos == std::string::npos)
    pos = op_name.length();

  const std::string sub = op_name.substr(0, pos);
  const at::Symbol m_symbol(at::Symbol::fromQualString(sub));
  return habana_lazy::is_inplace(m_symbol);
}

namespace detail {

void submit_result(at::Tensor& src, at::Tensor& result) {
  auto sizes = src.sizes().vec();
  if (src.sizes() != result.sizes()) {
    result.resize_(src.sizes());
  }

  result.copy_(src.to(result.scalar_type()));
}

at::Tensor& prepare_out(
    at::Tensor& from,
    at::Tensor& copy,
    at::ScalarType float_dtype) {
  if (!from.is_floating_point())
    return from;
  if (from.dtype() == float_dtype)
    return from;
  copy = at::empty(
      from.sizes(),
      float_dtype,
      from.layout(),
      from.device(),
      from.is_pinned(),
      {});
  return copy;
}
} // namespace detail

static void updateAtensorView(
    const at::Tensor& old_tensor,
    const at::Tensor& new_tensor) {
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
  auto hb_tensor = habana_lazy::GetHbLazyTensor(old_tensor);
  auto tensor_id = hb_tensor.getTensorUniqueId();
  if (context->viewContext.GetViewTableEntry(tensor_id) != nullptr) {
    habana_lazy::strided_insert_hpu_lazy(old_tensor, new_tensor);
  }
}

static void updateTensorViewIfNeeded(
    const c10::IValue& dst,
    const c10::IValue& src,
    std::string op_name) {
  if (isInplaceOp(op_name)) {
    if (src.isTensor() && dst.isTensor()) {
      updateAtensorView(dst.toTensor(), src.toTensor());
    }
    /*foreach op can't supported on view tensor due to current limitation
      of native fallback implementation.
      Below code is kept as of now for future reference.
    */
    /*else if (
       op_name.find("foreach") != std::string::npos && src.isTensorList() &&
       dst.isTensorList()) {
     auto src_list = src.toTensorList();
     auto dst_list = dst.toTensorList();
     TORCH_CHECK(
         src_list.size() == dst_list.size(),
         "src & dst tensorlist expected to be of same size");
     auto new_tensor = src_list.begin();
     auto old_tensor = dst_list.begin();
     while (new_tensor != src_list.end() && old_tensor != dst_list.end()) {
       updateAtensorView(*old_tensor, *new_tensor);
       new_tensor++;
       old_tensor++;
     }
    }*/
  }
}

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  PT_FALLBACK_TRACE
  const auto& op_name = c10::toString(op.operator_name());
  HpuFallbackHelper::get()->check_fallback_allowed(op_name);
  HpuFallbackHelper::get()->increment_count(op_name);
  auto old_tensor = stack->size() > 0 ? stack->at(0) : c10::nullopt;

  at::native::cpu_fallback(op, stack);

  auto new_tensor = stack->size() > 0 ? stack->at(0) : c10::nullopt;

  updateTensorViewIfNeeded(old_tensor, new_tensor, op_name);
}

TORCH_LIBRARY_IMPL(_, HPU, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace habana
