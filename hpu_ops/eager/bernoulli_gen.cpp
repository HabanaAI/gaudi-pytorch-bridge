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

#include "generated/eager/bernoulli.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
static void ScalarPToTensor(at::IValue& p, at::ScalarType dtype) {
  p = at::scalar_tensor(p.toScalar(), at::TensorOptions(at::kHPU).dtype(dtype));
}

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

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, BernoulliFE, at::Tensor&) {
  auto& p = get_inputs()[1];
  ScalarPToTensor(p, inputs[0].toTensor().scalar_type());

  ConvertGeneratorToSeedTensor(m_symbol, get_inputs().back());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, BernoulliOutFE, at::Tensor&) {
  auto& p = get_inputs()[1];
  ScalarPToTensor(p, inputs[0].toTensor().scalar_type());

  ConvertGeneratorToSeedTensor(m_symbol, get_inputs().rbegin()[1]);
}
} // namespace habana
