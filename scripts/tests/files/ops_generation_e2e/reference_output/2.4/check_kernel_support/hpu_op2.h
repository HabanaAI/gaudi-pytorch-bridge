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

static CheckNodeWithSharedLayerValidator validator_addbmm("addbmm", AddBMMSharedMeta);


struct shared_layer_as_strided : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 4) {
    auto ivalue_arr = torch::jit::last(stack, 4);
    if (ivalue_arr[0].isTensor() ) {

      c10::IValue self = std::move(peek(stack, 0, 4));
      c10::IValue size = std::move(peek(stack, 1, 4));
      c10::IValue stride = std::move(peek(stack, 2, 4));
      c10::IValue storage_offset = std::move(peek(stack, 3, 4));

      at::Tensor self_base = self.to<at::Tensor>();
      std::vector<int64_t> size_vec;
      const c10::List<c10::IValue> size_list_in = size.toList();

      for (c10::IValue size_elem: size_list_in) {
          int64_t size_elem_base = size_elem.to<int64_t>();
          size_vec.push_back(size_elem_base);
      }
      at::IntArrayRef size_list_out(size_vec);

      std::vector<int64_t> stride_vec;
      const c10::List<c10::IValue> stride_list_in = stride.toList();

      for (c10::IValue stride_elem: stride_list_in) {
          int64_t stride_elem_base = stride_elem.to<int64_t>();
          stride_vec.push_back(stride_elem_base);
      }
      at::IntArrayRef stride_list_out(stride_vec);


      auto storage_offset_opt = storage_offset.toOptional<c10::IValue>();
      ::std::optional<int64_t> storage_offset_opt_out;
      if (storage_offset_opt.has_value()) {
          const c10::IValue storage_offset_opt_in = storage_offset_opt.value();
          int64_t storage_offset_opt_in_base = storage_offset_opt_in.to<int64_t>();
          storage_offset_opt_out = ::std::optional<int64_t>(storage_offset_opt_in_base);
      } else {
          storage_offset_opt_out = ::std::optional<int64_t>();
      }

      auto is_supported = impl(self_base, size_list_out, stride_list_out, storage_offset_opt_out, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset, bool is_dynamic) {
  return true;
}

};

struct shared_layer_addbmm : SharedLayerOp {
bool func(torch::jit::Stack &stack, bool is_dynamic) {
  if (stack.size() == 5) {
    auto ivalue_arr = torch::jit::last(stack, 5);
    if (ivalue_arr[0].isTensor() && ivalue_arr[1].isTensor() && ivalue_arr[2].isTensor() && ivalue_arr[3].isScalar() && ivalue_arr[4].isScalar() ) {

      c10::IValue self = std::move(peek(stack, 0, 5));
      c10::IValue batch1 = std::move(peek(stack, 1, 5));
      c10::IValue batch2 = std::move(peek(stack, 2, 5));
      c10::IValue beta = std::move(peek(stack, 3, 5));
      c10::IValue alpha = std::move(peek(stack, 4, 5));

      at::Tensor self_base = self.to<at::Tensor>();
      at::Tensor batch1_base = batch1.to<at::Tensor>();
      at::Tensor batch2_base = batch2.to<at::Tensor>();
      at::Scalar beta_base = beta.to<at::Scalar>();
      at::Scalar alpha_base = alpha.to<at::Scalar>();
      auto is_supported = impl(self_base, batch1_base, batch2_base, beta_base, alpha_base, is_dynamic);
      return is_supported;
    }
  }
  return false;
}
private:
bool impl(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, bool is_dynamic) {
  VAL_CUSTOM_RETURN_IF_UNSUPPORTED_DTYPE(addbmm, is_dynamic, self, batch1, batch2, beta, alpha)

  return true;
}

};





}  // namespace habana

