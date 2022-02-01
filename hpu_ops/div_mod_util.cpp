/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "div_mod_util.h"

namespace habana {

std::shared_ptr<void> FillDivModParams(size_t& size, bool pyCompatible) {
  PARAMS_STUB(ns_DivModKernel::Params);
  // Python div_mod is enabled where remainder returns the same sign of the
  // divisor, except for the zero remainder, which is enforced by the default
  // value 'true' for pyCompatible. Other value (false) is used for div_rounding
  // mode operator, for 'trunc' case.
  params->isPyCompatible = pyCompatible;
  return params;
}

} // namespace habana