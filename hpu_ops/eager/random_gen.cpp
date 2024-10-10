/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "generated/eager/_fused_dropout.h"
#include "generated/eager/bernoulli.h"
#include "generated/eager/native_dropout.h"
#include "generated/eager/poisson.h"
#include "generated/eager/random.h"
#include "generated/eager/uniform.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
// Generators can't be represented in JIT graph
// https://github.com/pytorch/pytorch/issues/64005
static void ConvertGeneratorToSeedTensor(
    at::Symbol& symbol,
    at::IValue& gen_to_seed) {
  symbol = at::Symbol::fromQualString(
      "hpu::" + std::string(symbol.toUnqualString()));

  int seed = get_seed_hpu(gen_to_seed.toOptional<at::Generator>());
  at::TensorOptions o;
  o = o.dtype(at::kInt).device(at::kHPU);
  gen_to_seed = at::tensor(seed, o);
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, GeneratorToSeed, at::Tensor&) {
  ConvertGeneratorToSeedTensor(m_symbol, get_inputs().back());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, GeneratorToSeed, at::Tensor) {
  int stack_size = (int)get_inputs().size();
  int gen_pos = -1; // first position of generator or None type in the stack
  for (int i = 0; i < stack_size; i++) {
    if (get_inputs()[i].isNone() || get_inputs()[i].isGenerator()) {
      gen_pos = i;
      break;
    }
  }
  if (gen_pos != -1) {
    ConvertGeneratorToSeedTensor(m_symbol, get_inputs()[gen_pos]);
  } else {
    ConvertGeneratorToSeedTensor(m_symbol, get_inputs().back());
  }
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    eager::EagerOp,
    GeneratorToSeed,
    std::tuple<at::Tensor, at::Tensor>) {
  ConvertGeneratorToSeedTensor(m_symbol, get_inputs().back());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    eager::EagerOp,
    GeneratorToSeedOut,
    at::Tensor&) {
  ConvertGeneratorToSeedTensor(m_symbol, get_inputs().rbegin()[1]);
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
    eager::EagerOp,
    NativeDropoutFE,
    std::tuple<at::Tensor, at::Tensor>) {
  c10::IValue fakeGenToSeed = c10::optional<at::Generator>{};
  ConvertGeneratorToSeedTensor(m_symbol, fakeGenToSeed);
  get_inputs().back() = fakeGenToSeed;
}

} // namespace habana
