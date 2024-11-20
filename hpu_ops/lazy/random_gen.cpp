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

#include "backend/random.h"
#include "generated/lazy/_fused_dropout.h"
#include "generated/lazy/bernoulli.h"
#include "generated/lazy/native_dropout.h"

namespace habana {
// Generators can't be represented in JIT graph
// https://github.com/pytorch/pytorch/issues/64005
static void ConvertGeneratorToSeedTensor(at::IValue& gen_to_seed) {
  gen_to_seed = get_seed_tensor_hpu(gen_to_seed.toOptional<at::Generator>());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    GeneratorToSeed,
    at::Tensor&) {
  ConvertGeneratorToSeedTensor(get_inputs().back());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    GeneratorToSeed,
    at::Tensor) {
  int stack_size = (int)get_inputs().size();
  int gen_pos = -1; // first position of generator or None type in the stack
  for (int i = 0; i < stack_size; i++) {
    if (get_inputs()[i].isNone() || get_inputs()[i].isGenerator()) {
      gen_pos = i;
      break;
    }
  }
  if (gen_pos != -1) {
    ConvertGeneratorToSeedTensor(get_inputs()[gen_pos]);
  } else {
    ConvertGeneratorToSeedTensor(get_inputs().back());
  }
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    GeneratorToSeed,
    std::tuple<at::Tensor, at::Tensor>) {
  ConvertGeneratorToSeedTensor(get_inputs().back());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    GeneratorToSeedOut,
    at::Tensor&) {
  ConvertGeneratorToSeedTensor(get_inputs().rbegin()[1]);
}

unsigned NativeDropoutEarlyExitCondition(
    const at::Tensor& input,
    double p,
    c10::optional<bool> train) {
  if (input.numel() == 0) {
    return 1;
  }
  if ((train.has_value() && !*train) || (p == 0.0)) {
    return 2;
  }
  return 0;
}

::std::tuple<at::Tensor, at::Tensor> NativeDropoutEarlyExit(
    unsigned eePath,
    const at::Tensor& input,
    double,
    c10::optional<bool>) {
  if (eePath == 1) {
    return std::make_tuple(input, at::empty_like(input, input.options()));
  } else {
    return std::make_tuple(
        input.clone(),
        at::full(
            {1},
            1,
            {},
            c10::CppTypeToScalarType<bool>::value,
            input.options().layout_opt(),
            input.options().device_opt(),
            input.options().pinned_memory_opt())
            .expand(input.sizes()));
  }
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    NativeDropoutFE,
    std::tuple<at::Tensor, at::Tensor>) {
  c10::IValue fakeGenToSeed = c10::optional<at::Generator>{};
  ConvertGeneratorToSeedTensor(fakeGenToSeed);
  get_inputs().back() = fakeGenToSeed;
}

} // namespace habana
