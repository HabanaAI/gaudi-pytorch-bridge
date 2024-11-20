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
#include <pytorch_helpers/habana_helpers/pt_version_check.h>
#include "generated/backend/_prelu_kernel.h"

namespace habana {
OutputMetaDataVector PreluFwdMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.shape = input.sizes().vec();
  meta.dtype = input.scalar_type();

  return {meta};
}

OutputMetaDataVector PreluBwdMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 1);
  const auto& weight = stack_tensor(stack, 2);

  OutputMetaDataVector meta(2);
  meta.at(0).shape = input.sizes().vec();
  meta.at(0).dtype = input.scalar_type();

  meta.at(1).shape = weight.sizes().vec();
  meta.at(1).dtype = weight.scalar_type();
  return meta;
}

} // namespace habana
