/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/all.h"
#include "generated/any.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_op_helper.h"
#include "reduction_template.h"

constexpr float cmp_value = 0; // value to compare with the reduce sum result
namespace habana {

sizes_vec AllAnyOutputShape(const at::Stack&) {
  return {{}};
}

sizes_vec AllAnyDimOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim = stack.at(1).toInt();
  const bool keepdim = stack.at(2).toBool();

  return ReductionOutputShape(self, dim, keepdim);
}

template <>
AllAnyOutputType<at::Tensor>::AllAnyOutputType(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, 0) {
  set_scalar_type(at::kBool);
}

template <>
at::Tensor AllAnyOutputType<at::Tensor>::get_result_overrideable() {
  return {};
}

std::vector<synapse_helpers::tensor> AnyCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    const at::IntArrayRef dim,
    const bool keepdim,
    synapse_helpers::tensor& input_,
    const at::IntArrayRef outshape) {
  const auto& dtype = at::kFloat;
  std::unique_ptr<synapse_helpers::tensor> cast;
  synTensor& input = input_.get();
  if (dtype != self.scalar_type()) {
    cast = std::make_unique<synapse_helpers::tensor>(OpBackend::BuildCast(
        op, graph, input, self.sizes(), self.scalar_type(), dtype));
    input = cast->get();
  }

  op->SetScalarType(dtype);

  auto abs = OpBackend::BuildNode(
      op, graph, {"abs_fwd_f32", {input}, {{self.sizes().vec()}}});

  auto reduce_sum = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      {abs[0].get()},
      dim,
      keepdim,
      "reduce_sum_fwd_f32",
      {{outshape}});

  auto zero_tensor = OpBackend::BuildConstant(op, graph, cmp_value);

  return OpBackend::BuildNode(
      op,
      graph,
      {"greater_fwd_f32",
       {reduce_sum[0].get(), zero_tensor.get()},
       {{outshape, c10::ScalarType::Bool, 0}}});
}

void AnyDim::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  synapse_helpers::tensor& input = GetSynInputs()[0];
  auto outshape = ComputeOutputShapes(stack)[0];
  auto dim = stack.at(1).toInt();
  bool keepdim = stack.at(2).toBool();

  auto any_out =
      AnyCommonFunc(this, graph, self, dim, keepdim, input, outshape);
  syn_out(0) = std::move(any_out[0]);
}

void Any::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  synapse_helpers::tensor& input = GetSynInputs()[0];
  auto outshape = ComputeOutputShapes(stack)[0];

  auto any_out = AnyCommonFunc(this, graph, self, {}, false, input, outshape);
  syn_out(0) = std::move(any_out[0]);
}
} // namespace habana
