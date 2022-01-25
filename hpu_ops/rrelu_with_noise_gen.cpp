#include "generated/hpu_op.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
template <>
LazyRreluOutInplace<at::Tensor&>::LazyRreluOutInplace(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // out variant last argument is not a generator
  get_inputs().at(5) =
      get_seed_tensor_hpu(inputs.at(5).toOptional<at::Generator>());
}

template <>
at::Tensor& LazyRreluOutInplace<at::Tensor&>::get_result_overrideable() {
  return stack_tensor(get_inputs(), 0);
}

void Rrelu_with_noise::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto training = stack.at(4).toBool();
  auto lower = stack.at(2).toScalar().to<float>();
  auto upper = stack.at(3).toScalar().to<float>();
  size_t size = 0;
  if (training) {
    PARAMS_STUB(ns_RandomUniform::Params);
    params->low = lower;
    params->high = upper;
    // uniform random tensor
    auto uniform_random = BuildOp(
        graph,
        "random_uniform_fwd_" +
            habana_helpers::name_suffix_from_type(ScalarType()),
        {},
        {{outshape, ScalarType()}},
        params.get(),
        size);
    auto ones = ConstantHelper(graph, 1.0f, ScalarType(), outshape);
    auto zeros = ConstantHelper(graph, 0, ScalarType(), outshape);
    // cond: condition tensor
    auto cond = BuildOp(
        graph,
        "less_equal_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), zeros.get()},
        {{outshape, c10::ScalarType::Bool}});
    // noise: noise tensor
    auto noise = BuildOp(
        graph,
        "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {cond[0].get(), uniform_random[0].get(), ones.get()},
        {{outshape, ScalarType()}});
    // output
    auto output = BuildOp(
        graph,
        MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), noise[0].get()},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(output[0]);
  } else {
    PARAMS_STUB(ns_LeakyReluKernel::Params);
    auto negative_slope = (lower + upper) / 2;
    params->alpha = negative_slope;
    auto output = BuildOp(
        graph,
        "leakyrelu_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{outshape, ScalarType(), 0}},
        params.get(),
        size);
    syn_out(0) = std::move(output[0]);
  }
}

void Rrelu_with_noise_bwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto training = stack.at(5).toBool();
  auto lower = stack.at(3).toScalar().to<float>();
  auto upper = stack.at(4).toScalar().to<float>();
  if (training && (upper - lower) > 1e-6) {
    // grad_out * noise
    auto output = BuildOp(
        graph,
        MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(2)},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(output[0]);
  } else {
    size_t size = 0;
    PARAMS_STUB(ns_LeakyReluKernel::Params);
    auto negative_slope = (lower + upper) / 2;
    params->alpha = negative_slope;
    auto output = BuildOp(
        graph,
        "leakyrelu_bwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outshape, ScalarType(), 0}},
        params.get(),
        size);
    syn_out(0) = std::move(output[0]);
  }
}
} // namespace habana
