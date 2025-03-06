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
#pragma once
#include "aten_lazy_bridge.h"
#include "backend/synapse_helpers/env_flags.h"
#include "hpu_lazy_tensors.h"
#include "tensor_impl.h"

// namespace habana_lazy
namespace habana_lazy {

/**
 * @brief Class to provide utilities for dumping lazy-tensor graphs
 * in text/dot format.
 */
class IrGraphDumpUtil {
 public:
  static std::string ToDot(std::vector<ir::NodePtr> nodes);

  static std::string PostOrderToDot(
      const std::vector<ir::NodePtr>& post_order,
      const std::vector<ir::NodePtr>& roots,
      const bool use_ir_names = true);

  static std::string ToText(std::vector<ir::NodePtr> nodes);

  static std::string PostOrderToText(
      const std::vector<ir::NodePtr>& post_order,
      const std::vector<ir::NodePtr>& roots,
      const bool use_ir_names = true,
      const bool print_ir_graph_info = false);
};

} // namespace habana_lazy
