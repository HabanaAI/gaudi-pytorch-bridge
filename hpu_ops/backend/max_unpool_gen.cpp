/**
 * Copyright (c) 2021-2025 Intel Corporation
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

/**
 * Copyright (c) 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
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

#include "backend/habana_operator.h"
#include "generated/backend/max_unpool2d.h"
#include "generated/backend/max_unpool3d.h"
#include "pytorch_helpers/habana_helpers/conversion.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {

OutputMetaDataVector MaxUnpoolCommonMeta(
    const at::Stack& stack,
    const bool is_3d) {
  const auto self = stack.at(0).toTensor();
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype = self.scalar_type();
  meta.shape = self.sizes().vec();

  auto output_size = stack.at(2).toIntList();
  const size_t output_shape_dim = meta.shape.size();
  const size_t output_size_dim = output_size.size();

  meta.shape[output_shape_dim - 1] = output_size[output_size_dim - 1];
  meta.shape[output_shape_dim - 2] = output_size[output_size_dim - 2];
  if (is_3d) {
    meta.shape[output_shape_dim - 3] = output_size[output_size_dim - 3];
  }

  return metaVec;
}

OutputMetaDataVector MaxUnpool2DMeta(const at::Stack& stack) {
  return MaxUnpoolCommonMeta(stack, false);
}

OutputMetaDataVector MaxUnpool3DMeta(const at::Stack& stack) {
  return MaxUnpoolCommonMeta(stack, true);
}

FillParamsT FillMaxUnpool2DParams(const at::Stack& stack) {
  PARAMS_STUB(ns_MaxUnpool::Params);
  auto output_size = stack.at(2).toIntList();
  const size_t output_size_dim = output_size.size();

  params->output_h =
      safe_convert<unsigned int, int64_t>(output_size[output_size_dim - 1]);
  params->output_w =
      safe_convert<unsigned int, int64_t>(output_size[output_size_dim - 2]);

  return paramsT;
}

FillParamsT FillMaxUnpool3DParams(const at::Stack& stack) {
  PARAMS_STUB(ns_MaxUnpool::ParamsV2);
  auto output_size = stack.at(2).toIntList();
  const size_t output_size_dim = output_size.size();

  params->output_h =
      safe_convert<unsigned int, int64_t>(output_size[output_size_dim - 1]);
  params->output_w =
      safe_convert<unsigned int, int64_t>(output_size[output_size_dim - 2]);
  params->output_d =
      safe_convert<unsigned int, int64_t>(output_size[output_size_dim - 3]);

  return paramsT;
}

} // namespace habana
