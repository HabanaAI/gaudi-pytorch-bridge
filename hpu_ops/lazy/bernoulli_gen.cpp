/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/lazy/bernoulli.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
static void ConvertGeneratorToSeedTensor(at::IValue& gen_to_seed) {
  gen_to_seed = get_seed_tensor_hpu(gen_to_seed.toOptional<at::Generator>());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    BernoulliFE,
    at::Tensor&) {
  ConvertGeneratorToSeedTensor(get_inputs().back());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    BernoulliOutFE,
    at::Tensor&) {
  ConvertGeneratorToSeedTensor(get_inputs().rbegin()[1]);
}
} // namespace habana
