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
#include "generated/eager/bernoulli.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
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
  ConvertGeneratorToSeedTensor(m_symbol, get_inputs().back());
}
} // namespace habana
