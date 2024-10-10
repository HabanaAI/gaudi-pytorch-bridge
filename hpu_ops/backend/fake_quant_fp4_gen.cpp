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
  const std::map<c10::string_view, FakeQuantizeNf4IntermediateDtype>
      fp4_dtypes = {
          {"fp4_e2m1", FakeQuantizeNf4IntermediateDtype::FP_121},
          {"fp4_e3m0", FakeQuantizeNf4IntermediateDtype::FP_130}};

  const auto inter_dtype = fp4_dtypes.find(stack.at(3).to<c10::string_view>());
  TORCH_CHECK(
      inter_dtype != fp4_dtypes.end(),
      "Unrecognized FP4 dtype. Use naming \"fp4_eXmY\"");

  PARAMS_STUB(ns_FakeQuantizeFp4::Params);
  params->round = stack.at(1).toBool() ? FakeQuantizeNf4Round::FQ_SR_AND_SFTZ
                                       : FakeQuantizeNf4Round::FQ_RNE;
  params->axis = stack.at(2).toScalar().toInt();
  params->inter_dt = inter_dtype->second;
  return params;
}

} // namespace habana
