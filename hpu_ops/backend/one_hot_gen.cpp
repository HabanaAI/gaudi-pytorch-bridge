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
#include "generated/backend/one_hot.h"
#include "hpu_ops/hpu_op_helper.h"

namespace {
constexpr int64_t DEFAULT_NUM_OF_CLASSES = -1;
}

namespace habana {
int64_t calculateNumberOfClasses(const at::Stack& stack) {
  const auto num_classes = stack.at(1).toInt();

  HABANA_ASSERT(
      num_classes != DEFAULT_NUM_OF_CLASSES, "Number of classes cannot be -1");

  return num_classes;
}

FillParamsT FillOneHotParams(const at::Stack& stack) {
  PARAMS_STUB(ns_OneHotKernel::Params);
  params->axis = 0;
  params->depth = static_cast<int>(calculateNumberOfClasses(stack));
  params->on_value = 1.0F;
  params->off_value = 0.0F;

  return paramsT;
}

OutputMetaDataVector OneHotMeta(const at::Stack& stack) {
  auto input = stack_tensor(stack, 0);
  OutputMetaData meta;
  const auto num_classes = calculateNumberOfClasses(stack);
  meta.shape = input.sizes().vec();
  meta.shape.push_back(num_classes);
  meta.dtype = input.scalar_type();

  return {meta};
}

struct OneHot : OpBackend {
  OneHot(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "one_hot_fwd", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(OneHotMeta);
    SetFillParams(FillOneHotParams);
  }
};

} // namespace habana

static const auto& OneHotKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu::one_hot",
        habana::OneHot);
