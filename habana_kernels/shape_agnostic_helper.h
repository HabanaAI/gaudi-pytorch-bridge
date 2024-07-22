/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly
 * prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
