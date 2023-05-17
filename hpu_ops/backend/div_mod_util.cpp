/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "hpu_ops/div_mod_util.h"
#include "generated/backend/div.h"

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

std::vector<synapse_helpers::tensor> GetDivModOutput(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor syn_numerator,
    synTensor syn_denominator,
    bool pyCompatible,
    const std::vector<long int> shape_out,
    const at::ScalarType result_type,
    DIV_MODE_OUTPUT_TYPE t) {
  static_cast<void>(t);
  auto inputs = {syn_numerator, syn_denominator};
  size_t size;
  const auto& params = FillDivModParams(size, pyCompatible);

  std::vector<NodeAttr::NodeOutputAttr> node_output_attr = {
      {c10::IntArrayRef(shape_out.data(), shape_out.size()),
       op->ScalarType(),
       0},
      {c10::IntArrayRef(shape_out.data(), shape_out.size()), op->ScalarType()}};
  if (DIV_MODE_OUTPUT_TYPE::REMAINDER == t) {
    std::reverse(node_output_attr.begin(), node_output_attr.end());
  }

  auto output = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("div_mod_fwd", op->ScalarType()),
       std::move(inputs),
       node_output_attr,
       params.get(),
       size});
  return output;
}

} // namespace habana
