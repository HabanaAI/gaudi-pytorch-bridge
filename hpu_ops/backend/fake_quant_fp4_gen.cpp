/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/fake_quant_fp4.h"

namespace habana {
std::shared_ptr<void> FillFakeQuantFp4Params(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_FakeQuantizeFp4::Params);
  params->round = stack.at(1).toBool() ? FakeQuantizeNf4Round::FQ_SR_AND_SFTZ
                                       : FakeQuantizeNf4Round::FQ_RNE;
  params->axis = stack.at(2).toScalar().toInt();
  params->inter_dt = stack.at(3).toScalar().toInt() == 130
      ? FakeQuantizeNf4IntermediateDtype::FP_130
      : FakeQuantizeNf4IntermediateDtype::FP_121;
  return params;
}

} // namespace habana
