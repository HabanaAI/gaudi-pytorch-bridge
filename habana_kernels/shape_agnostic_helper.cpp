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
#include "shape_agnostic_helper.h"

namespace habana {

void HpuShapeAgnosticHelper::enumerate_shape_agnostic_unsupported_ops() {
  if (!op_set.empty()) {
    op_set.clear();
  }
  auto cache_map =
      habana::OptimizedJitGraphCache::GetOptimizedJitCache().get_m_cache_map();
  for (const auto& entry : cache_map) {
    // ops like optimizer do not support eager compiler when running in eager
    // mode and use graph compiler and recipe cache. Such ops should not be
    // tested for shape agnostic flow.
    if (!entry.second->get_is_shape_agnostic_supported()) {
      op_set.insert(entry.second->GetOpName());
    }
  }
}

size_t HpuShapeAgnosticHelper::get_jit_cache_size() const {
  auto cache_map =
      habana::OptimizedJitGraphCache::GetOptimizedJitCache().get_m_cache_map();
  return cache_map.size();
}

void HpuShapeAgnosticHelper::clear_jit_cache() const {
  habana::OptimizedJitGraphCache::GetOptimizedJitCache().Clear();
}

} // namespace habana
