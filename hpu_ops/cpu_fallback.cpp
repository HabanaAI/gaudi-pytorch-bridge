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
#include "habana_lazy/hpu_stage_submission.h"
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

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  PT_FALLBACK_TRACE
  const auto& op_name = c10::toString(op.operator_name());
  HpuFallbackHelper::get()->check_fallback_allowed(op_name);
  HpuFallbackHelper::get()->increment_count(op_name);

  auto& device = synapse_helpers::HPURegistrar::get_device();

  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());

  HABANA_ASSERT(
      context->getCapturing() == false,
      "cpu fallback is not supported during hpu graph capturing");

  at::native::cpu_fallback(op, stack);

  ::habana_lazy::StageSubmission::getInstance().setStageSubmissionFlow(
      ::habana_lazy::StageSubmission::Mode::SET_WHEN_CPU_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, HPU, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace habana
