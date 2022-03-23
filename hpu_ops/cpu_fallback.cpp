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

namespace habana {

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& op_name = c10::toString(op.operator_name());

  HpuFallbackHelper::get()->check_fallback_allowed(op_name);
  HpuFallbackHelper::get()->increment_count(op_name);

  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, HPU, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}
} // namespace habana
