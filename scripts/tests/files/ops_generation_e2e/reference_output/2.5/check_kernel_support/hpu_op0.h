// Autogenerated file by gen_op.py. Do not edit directly!

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/tensorexpr/tensorexpr_init.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include "cpu_fallback.h"

using habana_helpers::DTypeHelper;
using namespace torch::jit;


namespace habana {



struct shared_layer___ilshift__ : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 2) {
    auto ivalue_arr = torch::jit::last(stack, 2);
    if (ivalue_arr[0].isTensor() && ivalue_arr[1].isScalar() ) {

      c10::IValue self = std::move(peek(stack, 0, 2));
      c10::IValue other = std::move(peek(stack, 1, 2));

      at::Tensor self_base = self.to<at::Tensor>();
      at::Scalar other_base = other.to<at::Scalar>();
      auto is_supported = impl(self_base, other_base, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(at::Tensor & self, const at::Scalar & other, bool is_dynamic) {
  HPU_SUPPORTED_DTYPES(({{-1, {at::kInt, at::kChar, at::kByte, at::kShort, at::kBool}}}))
  RETURN_IF_UNSUPPORTED_DTYPE2(self, __ilshift__, is_dynamic, Scalar, self, other)

  return true;
}

};

struct shared_layer__foreach_add_ : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 2) {
    auto ivalue_arr = torch::jit::last(stack, 2);
    if (ivalue_arr[0].isTensorList() && ivalue_arr[1].isScalar() ) {

      c10::IValue self = std::move(peek(stack, 0, 2));
      c10::IValue scalar = std::move(peek(stack, 1, 2));

      std::vector<at::Tensor> self_vec;
      const c10::List<c10::IValue> self_list_in = self.toList();

      for (c10::IValue self_elem: self_list_in) {
          at::Tensor self_elem_base = self_elem.to<at::Tensor>();
          self_vec.push_back(self_elem_base);
      }
      at::TensorList self_list_out(self_vec);

      at::Scalar scalar_base = scalar.to<at::Scalar>();
      auto is_supported = impl(self_list_out, scalar_base, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(at::TensorList self, const at::Scalar & scalar, bool is_dynamic) {
  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kLong, at::kInt, at::kShort, at::kChar, at::kDouble, at::kBool}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kLong, at::kInt, at::kShort, at::kChar, at::kHalf, at::kDouble, at::kBool}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kLong, at::kInt, at::kShort, at::kChar, at::kHalf, at::kDouble, at::kBool}}}))

  return true;
}

};





}  // namespace habana

