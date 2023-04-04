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
#include "generated/eager/poisson.h"
#include "generated/eager/random.h"
#include "generated/eager/uniform.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
// Generators can't be represented in JIT graph
// https://github.com/pytorch/pytorch/issues/64005
static void ConvertGeneratorToSeedTensor(at::IValue& gen_to_seed) {
  int seed = get_seed_hpu(gen_to_seed.toOptional<at::Generator>());
  at::TensorOptions o;
  o = o.dtype(at::kInt).device(at::kHPU);
  gen_to_seed = at::tensor(seed, o);
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, GeneratorToSeed, at::Tensor&) {
  ConvertGeneratorToSeedTensor(get_inputs().back());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, GeneratorToSeed, at::Tensor) {
  ConvertGeneratorToSeedTensor(get_inputs().back());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    eager::EagerOp,
    GeneratorToSeed,
    std::tuple<at::Tensor, at::Tensor>) {
  ConvertGeneratorToSeedTensor(get_inputs().back());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    eager::EagerOp,
    GeneratorToSeedOut,
    at::Tensor&) {
  ConvertGeneratorToSeedTensor(get_inputs().rbegin()[1]);
}
} // namespace habana
