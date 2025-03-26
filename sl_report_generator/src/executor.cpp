/**
 * Copyright (c) 2024-2025 Intel Corporation
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

#include "executor.h"
namespace slrg {

template <typename T>
void SharedLayerExecutor<T>::validate() {
  if (validated)
    return;
  for (const auto precision_type : report_precision_types) {
    if (generator->isBlacklistedPrecisionType(precision_type)) {
      report[precision_type] = false;
    } else if (generator->isWhitelistedPrecisionType(precision_type)) {
      report[precision_type] = true;
    } else {
      const auto& stacks = generator->getStacks(precision_type);
      bool all_valid = true;
      for (const auto& stack : stacks) {
        try {
          all_valid &= checkNodeWithSharedLayer(stack);
        } catch (std::exception& e) {
          throw std::logic_error(
              "SharedLayerExecutor::validate failed for op '" +
              generator->getOpAndOverloadName() +
              "'. Possibly incorrect input quantity, signature or TORCH_CHECK inside output_meta.\nOriginal exception: " +
              e.what());
        }
      }
      report[precision_type] = all_valid;
    }
  }
  validated = true;
}

} // namespace slrg
