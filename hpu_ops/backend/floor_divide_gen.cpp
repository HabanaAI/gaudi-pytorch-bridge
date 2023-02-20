/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/floor_divide.h"
#include "habana_kernels/binary_kernels.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana {

std::shared_ptr<void> FillFloorDivideParams(
    const at::Stack& stack,
    size_t& size) {
  _TORCH_WARN_ONCE(
      "floor_divide is deprecated, and will be removed in a future version of pytorch."
      "It currently rounds toward 0 (like the \'trunc\' function NOT \'floor\')."
      "This results in incorrect rounding for negative values."
      "To keep the current behavior, use torch.div(a, b, rounding_mode=\'trunc\'),"
      "or for actual floor division, use torch.div(a, b, rounding_mode=\'floor\'). (function operator())");
  static_cast<void>(stack);
  PARAMS_STUB(ns_DivModKernel::ParamsV2);
#if IS_PYTORCH_OLDER_THAN(1, 13)
  params->isTruncRoundingMode = true;
#else
  params->isTruncRoundingMode = false;
#endif
  return params;
}

} // namespace habana
