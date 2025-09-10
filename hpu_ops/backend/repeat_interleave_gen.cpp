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

#include <perf_lib_layer_params.h>
#include "backend/habana_operator.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/repeat_interleave.h"
#include "pytorch_helpers/habana_helpers/conversion.h"
#include "pytorch_helpers/habana_helpers/logging.h"

using namespace std::literals;

namespace habana {

OutputMetaDataVector RepeatInterleaveMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto output_size_opt = stack.at(1).toOptional<int64_t>();
  HABANA_ASSERT(self.dim() == 1, "Self tensor is expected to be 1D.");
  HABANA_ASSERT(
      output_size_opt.has_value(),
      "It is expected that output_size is provided after frontend execution.");

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = std::vector<int64_t>{output_size_opt.value()};

  return {meta};
}

FillParamsT RepeatInterleaveParams(const at::Stack& stack) {
  PARAMS_STUB(ns_RepeatInterleave::Params);
  const auto opt_val = stack.at(1).toOptional<int64_t>();
  if (opt_val.has_value()) {
    params->outputSize =
        safe_convert<unsigned int>(opt_val.value(), "outputSize"sv);
  }

  return paramsT;
}

RepeatInterleave::RepeatInterleave(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "repeat_interleave_fwd",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(RepeatInterleaveMeta);
  SetFillParams(RepeatInterleaveParams);
}

} // namespace habana

static const auto& HabanaRepeatInterleaveKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "aten::repeat_interleave.Tensor",
        habana::RepeatInterleave);
