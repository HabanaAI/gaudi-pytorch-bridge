/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "cpu_fallback.h"
#include "backend/helpers/event_dispatcher.h"
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

auto to_cpu(const std::vector<c10::optional<at::Tensor>>& tensors) {
  std::vector<c10::optional<at::Tensor>> result(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    if (tensor && tensor->defined()) {
      result[i] = tensor->cpu();
    } else {
      result[i] = tensor;
    }
  }
  return result;
}

void convert_optional_tensor_lists_to_cpu(torch::jit::Stack* stack) {
  for (const auto idx : c10::irange(stack->size())) {
    const auto& ivalue = (*stack)[idx];
    if (ivalue.isOptionalTensorList()) {
      auto cpu_ivalue = c10::IValue(c10::List<c10::optional<at::Tensor>>(
          to_cpu(ivalue.toOptionalTensorList().vec())));
      (*stack)[idx] = std::move(cpu_ivalue);
    }
  }
}

} // namespace detail

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  PT_FALLBACK_TRACE
  const auto& op_name = c10::toString(op.operator_name());
  HpuFallbackHelper::get()->check_fallback_allowed(op_name);
  HpuFallbackHelper::get()->increment_count(op_name);
  habana_helpers::EmitEvent(
      habana_helpers::EventDispatcher::Topic::CPU_FALLBACK,
      habana_helpers::EventDispatcher::EventParams({{"op_name", op_name}}));


  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();

  HABANA_ASSERT(
      context->getCapturing() == false,
      "cpu fallback is not supported during hpu graph capturing");

  // Note: torch's native cpu_fallback doesn't handle optional tensor lists, so
  // they need to be moved to cpu here.
  detail::convert_optional_tensor_lists_to_cpu(stack);

  at::native::cpu_fallback(op, stack);

  ::habana_lazy::StageSubmission::getInstance().setStageSubmissionFlow(
      ::habana_lazy::StageSubmission::Mode::SET_WHEN_CPU_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, HPU, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace habana
