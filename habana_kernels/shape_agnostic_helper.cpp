/******************************************************************************
 * Copyright (C) 2021-2024 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly
 * prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "shape_agnostic_helper.h"

namespace habana {

void HpuShapeAgnosticHelper::enumerate_shape_agnostic_unsupported_ops() {
  if (!op_set.empty()) {
    op_set.clear();
  }
  auto cache_map =
      habana::OptimizedJitGraphCache::GetOptimizedJitCache().get_m_cache_map();
  for (const auto& entry : cache_map) {
    if (!entry.second->get_is_shape_agnostic_supported()) {
      op_set.insert(entry.second->GetOpName());
    }
  }
}

} // namespace habana
