#include <cstdint>
#include "generated/bitwise_and.h"
#include "generated/bitwise_or.h"
#include "generated/bitwise_xor.h"
#include "hpu_op_helper.h"

namespace habana {

static void bitwise_convert_scalar_to_tensor(
    std::vector<at::IValue>& inputs,
    const std::int64_t tensor_index,
    const std::int64_t scalar_index) {
  auto self = inputs[tensor_index].toTensor();
  auto other_tensor = habana_lazy::get_tensor_for_scalar(
      inputs[scalar_index].toScalar().toDouble(), self.scalar_type());
  inputs[scalar_index] = other_tensor;
}

template <typename T>
LazyBitwiseScalar<T>::LazyBitwiseScalar(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  auto input = LazyBitwiseScalar<T>::get_inputs();
  bitwise_convert_scalar_to_tensor(
      input, 0 /*tensor_index*/, 1 /*tensor_index*/);
  LazyBitwiseScalar<T>::set_inputs(input);
}

template struct LazyBitwiseScalar<at::Tensor&>;
template struct LazyBitwiseScalar<at::Tensor>;

template <typename T>
T LazyBitwiseScalar<T>::get_result_overrideable() {
  return LazyBitwiseScalar<T>::get_result_overrideable();
}

template <>
LazyBitwiseScalarTensor<at::Tensor>::LazyBitwiseScalarTensor(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  auto input = get_inputs();
  bitwise_convert_scalar_to_tensor(
      input, 1 /*tensor_index*/, 0 /*scalar_index*/);
  set_inputs(input);
}

template <>
at::Tensor LazyBitwiseScalarTensor<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(1).toTensor();
  return habana_lazy::empty_hpu_lazy(
      t.sizes(), t.options(), t.suggest_memory_format(), false);
}
} // namespace habana