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

static CheckNodeWithSharedLayerValidator validator_bucketize_Scalar("bucketize.Scalar", "search_sorted_fwd", {1}, {0}, BucketizeMeta, {0, 1}, false, false, false, false);
static CheckNodeWithSharedLayerValidator validator_elu("elu", "elu_fwd", {0}, {}, nullptr, {}, false, false, false, false);


struct shared_layer_bucketize : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 4) {
    auto ivalue_arr = torch::jit::last(stack, 4);
    if (ivalue_arr[0].isScalar() && ivalue_arr[1].isTensor() && ivalue_arr[2].isBool() && ivalue_arr[3].isBool() ) {

      c10::IValue self = std::move(peek(stack, 0, 4));
      c10::IValue boundaries = std::move(peek(stack, 1, 4));
      c10::IValue out_int32 = std::move(peek(stack, 2, 4));
      c10::IValue right = std::move(peek(stack, 3, 4));

      at::Scalar self_base = self.to<at::Scalar>();
      at::Tensor boundaries_base = boundaries.to<at::Tensor>();
      bool out_int32_base = out_int32.to<bool>();
      bool right_base = right.to<bool>();
      auto is_supported = impl(self_base, boundaries_base, out_int32_base, right_base, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right, bool is_dynamic) {
  auto compute_type = DTypeHelper::get_compute_dtype({self, boundaries}, c10::nullopt, DTypeHelper::DtypePromoteVariant::kPromoteToCommon, false/*safe_cast*/);
  static_cast<void>(compute_type);

  VAL_RETURN_IF_UNSUPPORTED_DTYPE2(bucketize, is_dynamic, Scalar, self, boundaries, out_int32, right)

  return true;
}

};

struct shared_layer_elu : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 4) {
    auto ivalue_arr = torch::jit::last(stack, 4);
    if (ivalue_arr[0].isTensor() && ivalue_arr[1].isScalar() && ivalue_arr[2].isScalar() && ivalue_arr[3].isScalar() ) {

      c10::IValue self = std::move(peek(stack, 0, 4));
      c10::IValue alpha = std::move(peek(stack, 1, 4));
      c10::IValue scale = std::move(peek(stack, 2, 4));
      c10::IValue input_scale = std::move(peek(stack, 3, 4));

      at::Tensor self_base = self.to<at::Tensor>();
      at::Scalar alpha_base = alpha.to<at::Scalar>();
      at::Scalar scale_base = scale.to<at::Scalar>();
      at::Scalar input_scale_base = input_scale.to<at::Scalar>();
      auto is_supported = impl(self_base, alpha_base, scale_base, input_scale_base, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_dynamic) {
  VAL_RETURN_IF_UNSUPPORTED_DTYPE(elu, is_dynamic, self, alpha, scale, input_scale)

  return true;
}

};





}  // namespace habana

