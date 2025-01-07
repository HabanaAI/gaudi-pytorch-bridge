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



struct shared_layer_convolution_backward_overrideable : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 10) {
    auto ivalue_arr = torch::jit::last(stack, 10);
    if (ivalue_arr[0].isTensor() && ivalue_arr[1].isTensor() && ivalue_arr[2].isTensor() && ivalue_arr[6].isBool() ) {

      c10::IValue grad_output = std::move(peek(stack, 0, 10));
      c10::IValue input = std::move(peek(stack, 1, 10));
      c10::IValue weight = std::move(peek(stack, 2, 10));
      c10::IValue stride = std::move(peek(stack, 3, 10));
      c10::IValue padding = std::move(peek(stack, 4, 10));
      c10::IValue dilation = std::move(peek(stack, 5, 10));
      c10::IValue transposed = std::move(peek(stack, 6, 10));
      c10::IValue output_padding = std::move(peek(stack, 7, 10));
      c10::IValue groups = std::move(peek(stack, 8, 10));
      c10::IValue output_mask = std::move(peek(stack, 9, 10));

      at::Tensor grad_output_base = grad_output.to<at::Tensor>();
      at::Tensor input_base = input.to<at::Tensor>();
      at::Tensor weight_base = weight.to<at::Tensor>();
      std::vector<int64_t> stride_vec;
      const c10::List<c10::IValue> stride_list_in = stride.toList();

      for (c10::IValue stride_elem: stride_list_in) {
          int64_t stride_elem_base = stride_elem.to<int64_t>();
          stride_vec.push_back(stride_elem_base);
      }
      at::IntArrayRef stride_list_out(stride_vec);

      std::vector<int64_t> padding_vec;
      const c10::List<c10::IValue> padding_list_in = padding.toList();

      for (c10::IValue padding_elem: padding_list_in) {
          int64_t padding_elem_base = padding_elem.to<int64_t>();
          padding_vec.push_back(padding_elem_base);
      }
      at::IntArrayRef padding_list_out(padding_vec);

      std::vector<int64_t> dilation_vec;
      const c10::List<c10::IValue> dilation_list_in = dilation.toList();

      for (c10::IValue dilation_elem: dilation_list_in) {
          int64_t dilation_elem_base = dilation_elem.to<int64_t>();
          dilation_vec.push_back(dilation_elem_base);
      }
      at::IntArrayRef dilation_list_out(dilation_vec);

      bool transposed_base = transposed.to<bool>();
      std::vector<int64_t> output_padding_vec;
      const c10::List<c10::IValue> output_padding_list_in = output_padding.toList();

      for (c10::IValue output_padding_elem: output_padding_list_in) {
          int64_t output_padding_elem_base = output_padding_elem.to<int64_t>();
          output_padding_vec.push_back(output_padding_elem_base);
      }
      at::IntArrayRef output_padding_list_out(output_padding_vec);

      int64_t groups_base = groups.to<int64_t>();
      const c10::List<c10::IValue> output_mask_list_in = output_mask.toList();

      ::std::array<bool,3> output_mask_list_out = as_array<bool, 3>(output_mask_list_in);

      auto is_supported = impl(grad_output_base, input_base, weight_base, stride_list_out, padding_list_out, dilation_list_out, transposed_base, output_padding_list_out, groups_base, output_mask_list_out, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask, bool is_dynamic) {
  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}}}))
  RETURN_IF_UNSUPPORTED_DTYPE(grad_output, convolution_backward_overrideable, is_dynamic, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask)
  RETURN_IF_UNSUPPORTED_DTYPE(input, convolution_backward_overrideable, is_dynamic, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask)
  RETURN_IF_UNSUPPORTED_DTYPE(weight, convolution_backward_overrideable, is_dynamic, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask)

  return true;
}

};

struct shared_layer_native_group_norm : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 8) {
    auto ivalue_arr = torch::jit::last(stack, 8);
    if (ivalue_arr[0].isTensor() && ivalue_arr[6].isInt() && ivalue_arr[7].isDouble() ) {

      c10::IValue input = std::move(peek(stack, 0, 8));
      c10::IValue weight = std::move(peek(stack, 1, 8));
      c10::IValue bias = std::move(peek(stack, 2, 8));
      c10::IValue N = std::move(peek(stack, 3, 8));
      c10::IValue C = std::move(peek(stack, 4, 8));
      c10::IValue HxW = std::move(peek(stack, 5, 8));
      c10::IValue group = std::move(peek(stack, 6, 8));
      c10::IValue eps = std::move(peek(stack, 7, 8));

      at::Tensor input_base = input.to<at::Tensor>();

      auto weight_opt = weight.toOptional<c10::IValue>();
      ::std::optional<at::Tensor> weight_opt_out;
      if (weight_opt.has_value()) {
          const c10::IValue weight_opt_in = weight_opt.value();
          at::Tensor weight_opt_in_base = weight_opt_in.to<at::Tensor>();
          weight_opt_out = ::std::optional<at::Tensor>(weight_opt_in_base);
      } else {
          weight_opt_out = ::std::optional<at::Tensor>();
      }


      auto bias_opt = bias.toOptional<c10::IValue>();
      ::std::optional<at::Tensor> bias_opt_out;
      if (bias_opt.has_value()) {
          const c10::IValue bias_opt_in = bias_opt.value();
          at::Tensor bias_opt_in_base = bias_opt_in.to<at::Tensor>();
          bias_opt_out = ::std::optional<at::Tensor>(bias_opt_in_base);
      } else {
          bias_opt_out = ::std::optional<at::Tensor>();
      }

      int64_t N_base = N.to<int64_t>();
      int64_t C_base = C.to<int64_t>();
      int64_t HxW_base = HxW.to<int64_t>();
      int64_t group_base = group.to<int64_t>();
      double eps_base = eps.to<double>();
      auto is_supported = impl(input_base, weight_opt_out, bias_opt_out, N_base, C_base, HxW_base, group_base, eps_base, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps, bool is_dynamic) {
  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kDouble}}}))
  RETURN_IF_UNSUPPORTED_DTYPE(input, native_group_norm, is_dynamic, input, weight, bias, N, C, HxW, group, eps)

  return true;
}

};

struct shared_layer_linear_backward : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 4) {
    auto ivalue_arr = torch::jit::last(stack, 4);
    if (ivalue_arr[0].isTensor() && ivalue_arr[1].isTensor() && ivalue_arr[2].isTensor() ) {

      c10::IValue self = std::move(peek(stack, 0, 4));
      c10::IValue grad_output = std::move(peek(stack, 1, 4));
      c10::IValue weight = std::move(peek(stack, 2, 4));
      c10::IValue output_mask = std::move(peek(stack, 3, 4));

      at::Tensor self_base = self.to<at::Tensor>();
      at::Tensor grad_output_base = grad_output.to<at::Tensor>();
      at::Tensor weight_base = weight.to<at::Tensor>();
      const c10::List<c10::IValue> output_mask_list_in = output_mask.toList();

      ::std::array<bool,3> output_mask_list_out = as_array<bool, 3>(output_mask_list_in);

      auto is_supported = impl(self_base, grad_output_base, weight_base, output_mask_list_out, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, ::std::array<bool,3> output_mask, bool is_dynamic) {
  HPU_SUPPORTED_DTYPES(({{synDeviceGaudi, {at::kBFloat16, at::kFloat, at::kDouble}},
   {synDeviceGaudi2, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}},
   {synDeviceGaudi3, {at::kBFloat16, at::kFloat, at::kHalf, at::kDouble}}}))
  RETURN_IF_UNSUPPORTED_DTYPE(self, linear_backward, is_dynamic, self, grad_output, weight, output_mask)
  RETURN_IF_UNSUPPORTED_DTYPE(grad_output, linear_backward, is_dynamic, self, grad_output, weight, output_mask)
  RETURN_IF_UNSUPPORTED_DTYPE(weight, linear_backward, is_dynamic, self, grad_output, weight, output_mask)

  return true;
}

};





}  // namespace habana

