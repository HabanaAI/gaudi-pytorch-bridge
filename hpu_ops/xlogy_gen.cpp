/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/special_xlog1py.h"
#include "generated/xlogy.h"

namespace habana {

constexpr size_t index_of_self = 0;
constexpr size_t index_of_other = 1;

static void XlogyScalarConversion(
    std::vector<at::IValue>& inputs,
    size_t scalar_index,
    size_t tensor_index) {
  auto tensor = inputs[tensor_index].toTensor();
  auto scalar = inputs[scalar_index].toScalar();
  auto dtype = at::result_type(tensor, scalar);
  auto self_tensor =
      habana_lazy::get_tensor_for_scalar(scalar.toDouble(), dtype);
  inputs[scalar_index] = c10::IValue(self_tensor);
}

template <typename T>
LazyXlogY<T>::LazyXlogY(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  auto x = LazyXlogY<T>::get_inputs();
  // convert scalar input to tensor
  if (x[index_of_self].isScalar()) {
    XlogyScalarConversion(x, index_of_self, index_of_other);
  } else {
    XlogyScalarConversion(x, index_of_other, index_of_self);
  }
  LazyXlogY<T>::set_inputs(x);
}

template struct LazyXlogY<at::Tensor&>;
template struct LazyXlogY<at::Tensor>;

template <typename T>
T LazyXlogY<T>::get_result_overrideable() {
  return LazyXlogY<T>::get_result_overrideable();
}

sizes_vec XlogYOutputShape(const at::Stack& stack, bool) {
  if (stack.at(1).isScalar()) {
    const torch::Tensor& self = stack_tensor(stack, 0);
    return {self.sizes().vec()};
  } else if (stack.at(0).isScalar()) {
    const torch::Tensor& other = stack_tensor(stack, 1);
    return {other.sizes().vec()};
  }
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& other = stack_tensor(stack, 1);
  return {at::infer_size(self.sizes(), other.sizes())};
}

void XlogYOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = XlogYOutputShape(stack, true)[0];
  auto other_shape = stack_tensor(stack, 1).sizes().vec();

  auto logy = BuildOp(graph, guid_, {syn_in(1)}, {{other_shape, ScalarType()}});
  auto xlogy = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), logy[0].get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(xlogy[0]);
}
} // namespace habana
