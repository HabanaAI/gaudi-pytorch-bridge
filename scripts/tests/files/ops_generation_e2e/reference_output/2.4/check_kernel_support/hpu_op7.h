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



struct shared_layer_isfinite : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 1) {
    auto ivalue_arr = torch::jit::last(stack, 1);
    if (ivalue_arr[0].isTensor() ) {

      c10::IValue self = std::move(peek(stack, 0, 1));

      at::Tensor self_base = self.to<at::Tensor>();
      auto is_supported = impl(self_base, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(const at::Tensor & self, bool is_dynamic) {
  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kInt, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kInt, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kInt, at::kDouble}}}))
  RETURN_IF_UNSUPPORTED_DTYPE(self, isfinite, is_dynamic, self)

  return true;
}

};

struct shared_layer_upsample_bicubic2d : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 4) {
    auto ivalue_arr = torch::jit::last(stack, 4);
    if (ivalue_arr[0].isTensor() && ivalue_arr[2].isBool() && ivalue_arr[3].isList() ) {

      c10::IValue input = std::move(peek(stack, 0, 4));
      c10::IValue output_size = std::move(peek(stack, 1, 4));
      c10::IValue align_corners = std::move(peek(stack, 2, 4));
      c10::IValue scale_factors = std::move(peek(stack, 3, 4));

      at::Tensor input_base = input.to<at::Tensor>();
      std::vector<int64_t> output_size_opt_in_vec;

      auto output_size_opt = output_size.toOptional<c10::IValue>();
      at::OptionalIntArrayRef output_size_opt_out;
      if (output_size_opt.has_value()) {
          const c10::IValue output_size_opt_in = output_size_opt.value();
          const c10::List<c10::IValue> output_size_opt_in_list_in = output_size_opt_in.toList();

        for (c10::IValue output_size_opt_in_elem: output_size_opt_in_list_in) {
            int64_t output_size_opt_in_elem_base = output_size_opt_in_elem.to<int64_t>();
            output_size_opt_in_vec.push_back(output_size_opt_in_elem_base);
        }
        at::IntArrayRef output_size_opt_in_list_out(output_size_opt_in_vec);

          output_size_opt_out = at::OptionalIntArrayRef(output_size_opt_in_list_out);
      } else {
          output_size_opt_out = at::OptionalIntArrayRef();
      }

      bool align_corners_base = align_corners.to<bool>();
      std::vector<double> scale_factors_opt_in_vec;

      auto scale_factors_opt = scale_factors.toOptional<c10::IValue>();
      ::std::optional<at::ArrayRef<double>> scale_factors_opt_out;
      if (scale_factors_opt.has_value()) {
          const c10::IValue scale_factors_opt_in = scale_factors_opt.value();
          const c10::List<c10::IValue> scale_factors_opt_in_list_in = scale_factors_opt_in.toList();

        for (c10::IValue scale_factors_opt_in_elem: scale_factors_opt_in_list_in) {
            double scale_factors_opt_in_elem_base = scale_factors_opt_in_elem.to<double>();
            scale_factors_opt_in_vec.push_back(scale_factors_opt_in_elem_base);
        }
        at::ArrayRef<double> scale_factors_opt_in_list_out(scale_factors_opt_in_vec);

          scale_factors_opt_out = ::std::optional<at::ArrayRef<double>>(scale_factors_opt_in_list_out);
      } else {
          scale_factors_opt_out = ::std::optional<at::ArrayRef<double>>();
      }

      auto is_supported = impl(input_base, output_size_opt_out, align_corners_base, scale_factors_opt_out, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(const at::Tensor & input, at::OptionalIntArrayRef output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors, bool is_dynamic) {
  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}}}))
  RETURN_IF_UNSUPPORTED_DTYPE2(input, upsample_bicubic2d, is_dynamic, vec, input, output_size, align_corners, scale_factors)

  return true;
}

};





}  // namespace habana

