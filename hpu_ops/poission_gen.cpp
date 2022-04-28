
/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"

namespace habana {
std::shared_ptr<void> FillPoissonParams(const at::Stack&, size_t& size) {
  PARAMS_STUB(ns_RandomPoisson::Params);
  params->lambda = 0.0;
  params->poissonFlavor = RandomPoissonFlavor_t::WITH_DIST;
  return params;
}

} // namespace habana
