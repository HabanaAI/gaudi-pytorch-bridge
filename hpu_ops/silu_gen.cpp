#include "generated/silu.h"
#include "generated/silu_backward.h"
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

void SiluBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  constexpr size_t idxGrad = 0;
  constexpr size_t idxSelf = 1;
  const auto& outshape = stack_tensor(stack, idxGrad).sizes();
  auto guidSuffix = habana_helpers::name_suffix_from_type(ScalarType());
  const std::string guidFwd = "_fwd_";

  // S=Sigmoid (self)
  auto sigmoid = BuildOp(
      graph,
      "sigmoid" + guidFwd + guidSuffix,
      {syn_in(idxSelf)},
      {{outshape, ScalarType()}});

  // G*S
  auto mul = BuildOp(
      graph,
      MULT_GUID + guidSuffix,
      {syn_in(idxGrad), sigmoid[0].get()},
      {{outshape, ScalarType()}});

  // self*G*S
  auto mul1 = BuildOp(
      graph,
      MULT_GUID + guidSuffix,
      {syn_in(idxSelf), mul[0].get()},
      {{outshape, ScalarType()}});

  // S*G*S*self
  auto mul2 = BuildOp(
      graph,
      MULT_GUID + guidSuffix,
      {sigmoid[0].get(), mul1[0].get()},
      {{outshape, ScalarType()}});

  // G*S + G*self*S
  auto add = BuildOp(
      graph,
      "add" + guidFwd + guidSuffix,
      {mul[0].get(), mul1[0].get()},
      {{outshape, ScalarType()}});

  // G*S + G*self*S - G*self*S*S
  auto sub = BuildOp(
      graph,
      "sub" + guidFwd + guidSuffix,
      {add[0].get(), mul2[0].get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(sub[0]);
}
} // namespace habana
