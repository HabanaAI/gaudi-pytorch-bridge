/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/ne.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

SharedMetaDataVector CompareNeSharedMeta(const at::Stack& stack) {
  auto equalSharedMetaVec = CompareSharedMeta(stack, "equal_fwd");
  SharedMetaData notSharedMeta{"not_fwd"};
  notSharedMeta.inputs_data = {equalSharedMetaVec[0].outputs_data[0]};
  notSharedMeta.outputs_data = {notSharedMeta.inputs_data[0]};
  equalSharedMetaVec.push_back(notSharedMeta);
  return equalSharedMetaVec;
}

void NE::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  auto outshape = stack[1].isScalar()
      ? self.sizes().vec()
      : at::infer_size(self.sizes(), stack_tensor(stack, 1).sizes());

  const at::ScalarType& result_type = c10::ScalarType::Bool;

  auto eq = BuildOp(
      graph,
      get_guid_with_precision("equal_fwd", ScalarType()),
      {syn_in(0), syn_in(1)},
      {{outshape, result_type}});

  // not on output of equal
  auto not_equal =
      BuildOp(graph, "not_fwd_i8", {eq[0].get()}, {{outshape, result_type, 0}});

  // output of not is the output of this op
  syn_out(0) = std::move(not_equal[0]);
}
} // namespace habana
