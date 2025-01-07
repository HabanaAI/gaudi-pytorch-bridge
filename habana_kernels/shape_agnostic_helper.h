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
#include <c10/util/Exception.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "backend/jit_graph_cache.h"
#include "habana_eager/eager_exec.h"
#include "habana_helpers/logging_pt.h"

#pragma once
namespace habana {
class HpuShapeAgnosticHelper {
 private:
  HpuShapeAgnosticHelper() {}

  ~HpuShapeAgnosticHelper() = default;

 public:
  static HpuShapeAgnosticHelper* get() {
    static HpuShapeAgnosticHelper* singleton = new HpuShapeAgnosticHelper();
    return singleton;
  }

  void enumerate_shape_agnostic_unsupported_ops();

  size_t get_jit_cache_size() const;
  void clear_jit_cache() const;

  const std::set<std::string>& get_shape_agnostic_unsupported_ops() {
    enumerate_shape_agnostic_unsupported_ops();
    return op_set;
  }

  const std::unordered_set<std::string_view>&
  get_eager_compiler_unsupported_op_prefixes() {
    return habana::OptimizedJitGraphCache::GetOptimizedJitCache()
        .get_eager_compiler_unsupported_op_prefixes();
  };

 private:
  std::set<std::string> op_set;
};

} // namespace habana
