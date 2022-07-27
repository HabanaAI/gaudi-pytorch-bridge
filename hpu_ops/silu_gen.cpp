#include "generated/hpu_op.h"
#include "hpu_op_helper.h"

namespace habana {
void Silu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto sigmoid = BuildOp(
      graph,
      "sigmoid_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType()}});

  auto mul = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {sigmoid[0].get(), syn_in(0)},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(mul[0]);
}
} // namespace habana
