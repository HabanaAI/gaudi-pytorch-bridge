/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/resize.h"

using namespace torch;

Tensor relu_hpu(const Tensor& input) {
  LOG_FUNC_BEGIN;
  auto output = at::empty(input.sizes(), input.options());
  std::vector<const at::Tensor*> pt_outputs{&output};
  std::vector<const at::Tensor*> pt_inputs{&input};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "relu", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return output;
}

Tensor& relu_hpu_(Tensor& self) {
  LOG_FUNC_BEGIN;
  std::vector<const at::Tensor*> pt_inputs{&self};

  synapse_simple_generic_inplace_kernel(
      pt_inputs, "relu", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.sigmoid(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor sigmoid_hpu(const Tensor& input) {
  LOG_FUNC_BEGIN;

  auto output = at::empty(input.sizes(), input.options());
  std::vector<const at::Tensor*> pt_outputs{&output};
  std::vector<const at::Tensor*> pt_inputs{&input};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "sigmoid",
      nullptr,
      0,
      SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.sigmoid(grad_in, input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] grad_in - input tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor sigmoid_backward_hpu(const Tensor& grad_in, const Tensor& input) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(
      grad_in.scalar_type() == input.scalar_type(),
      "Types don't match. grad_in type: ",
      grad_in.scalar_type(),
      " input type: ",
      input.scalar_type());
  TORCH_CHECK(
      (grad_in.sizes() == input.sizes()) ||
          (grad_in.ndimension() == input.ndimension() &&
           std::all_of(
               input.sizes().cbegin(),
               input.sizes().cend(),
               [](auto val) { return val == 1; })),
      "Sizes in elementwise kernel don't match. grad_in sizes: ",
      grad_in.sizes(),
      ", input sizes: ",
      grad_in.sizes());

  auto grad_output = at::empty(input.sizes(), input.options());
  std::vector<const at::Tensor*> pt_outputs{&grad_output};
  std::vector<const at::Tensor*> pt_inputs{&grad_in, &input};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "sigmoid",
      nullptr,
      0,
      SynapsePassType::BACKWARD_PASS);

  LOG_FUNC_END;
  return grad_output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.sqrt(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor sqrt_hpu(const Tensor& input) {
  LOG_FUNC_BEGIN;

  auto output = at::empty(input.sizes(), input.options());
  std::vector<const at::Tensor*> pt_outputs{&output};
  std::vector<const at::Tensor*> pt_inputs{&input};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "sqrt", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.tanh(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor tanh_hpu(const Tensor& input) {
  LOG_FUNC_BEGIN;

  auto output = at::empty(input.sizes(), input.options());
  std::vector<const at::Tensor*> pt_outputs{&output};
  std::vector<const at::Tensor*> pt_inputs{&input};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "tanh", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = a.tanh(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/

Tensor& tanh_hpu_(Tensor& self) {
  LOG_FUNC_BEGIN;
  std::vector<const at::Tensor*> pt_inputs{&self};

  synapse_simple_generic_inplace_kernel(
      pt_inputs, "tanh", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for torch.tanh(input,out)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/

Tensor& tanh_out_hpu(Tensor& out, Tensor& self) {
  LOG_FUNC_BEGIN;
  std::vector<const at::Tensor*> pt_inputs{&self};
  std::vector<const at::Tensor*> pt_outputs{&out};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "tanh", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.tanh(grad_in, input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] grad_in - input tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor tanh_backward_hpu(const Tensor& grad_in, const Tensor& input) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(
      grad_in.scalar_type() == input.scalar_type(),
      "Types don't match. grad_in type: ",
      grad_in.scalar_type(),
      " input type: ",
      input.scalar_type());
  TORCH_CHECK(
      (grad_in.sizes() == input.sizes()) ||
          (grad_in.ndimension() == input.ndimension() &&
           std::all_of(
               input.sizes().cbegin(),
               input.sizes().cend(),
               [](auto val) { return val == 1; })),
      "Sizes in elementwise kernel don't match. grad_in sizes: ",
      grad_in.sizes(),
      ", input sizes: ",
      grad_in.sizes());

  auto grad_output = at::empty(input.sizes(), input.options());
  std::vector<const at::Tensor*> pt_outputs{&grad_output};
  std::vector<const at::Tensor*> pt_inputs{&grad_in, &input};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "tanh",
      nullptr,
      0,
      SynapsePassType::BACKWARD_PASS);

  LOG_FUNC_END;
  return grad_output;
}

/*************************************************************************
 * @brief Kernel implementation for gelu
 *output = 0.5 * x *(1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor gelu_hpu(const Tensor& self) {
  LOG_FUNC_BEGIN;

  auto output = at::pow(self, 3.0);
  output = at::add(self, output, 0.044715);
  output.mul_(M_2_SQRTPI * M_SQRT1_2).tanh_().add_(1.0).mul_(0.5).mul_(self);

  LOG_FUNC_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for erf_
 * output = x.erf_()
 * erf(x) = tanh((2/sqrt(pi))*(x+0.08943*x^3))
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& erf_hpu_(Tensor& self) {
  LOG_FUNC_BEGIN;

  Tensor self_copy = at::empty(self.sizes(), self.options());
  habana_helpers::copy_data_within_device(self, self_copy);

  self.pow_(3.0).mul_(0.08943).add_(self_copy).mul_(M_2_SQRTPI).tanh_();

  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for exp_
 * output = x.exp_()
 * @param [out, in]  1-4D, BF16/FP32
 ************************************************************************/
Tensor& exp_hpu_(Tensor& self) {
  LOG_FUNC_BEGIN;

  auto self_copy = at::empty(self.sizes(), self.options());
  habana_helpers::copy_data_within_device(self, self_copy);

  std::vector<const at::Tensor*> pt_outputs{&self};
  std::vector<const at::Tensor*> pt_inputs{&self_copy};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "exp", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for torch.neg(input,out)
 * @param [out] out - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& neg_out_hpu( Tensor& result, const Tensor& input) {
  LOG_FUNC_BEGIN;

  // Resize result to correct size (if required)
  auto shape = DimVector(input.sizes());
  auto tht_result = result.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);

  std::vector<const at::Tensor*> pt_outputs{&result};
  std::vector<const at::Tensor*> pt_inputs{&input};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "neg", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return result;

}


/*************************************************************************
 * @brief Kernel implementation for inplace torch.reciprocal_(self)
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& reciprocal_hpu_(Tensor& self) {
  LOG_FUNC_BEGIN;
  std::vector<const at::Tensor*> pt_inputs{&self};

  synapse_simple_generic_inplace_kernel(
      pt_inputs, "reciprocal", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.reciprocal(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor reciprocal_hpu(const Tensor& self) {
  LOG_FUNC_BEGIN;

  auto output = at::empty(self.sizes(), self.options());
  std::vector<const at::Tensor*> pt_outputs{&output};
  std::vector<const at::Tensor*> pt_inputs{&self};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "reciprocal", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for torch.reciprocal(self,out)
 * @param [out] out - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& reciprocal_out_hpu( Tensor& result, const Tensor& self) {
  LOG_FUNC_BEGIN;

  // Resize result to correct size (if required)
  auto shape = DimVector(self.sizes());
  auto tht_result = result.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);

  std::vector<const at::Tensor*> pt_outputs{&result};
  std::vector<const at::Tensor*> pt_inputs{&self};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "reciprocal", nullptr, 0, SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return result;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema("aten::relu_(Tensor(a!) self) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(relu_hpu_), &relu_hpu_>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::relu(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(relu_hpu), &relu_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::sigmoid(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(sigmoid_hpu), &sigmoid_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(sigmoid_backward_hpu),
                    &sigmoid_backward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::sqrt(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(sqrt_hpu), &sqrt_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::tanh(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(tanh_hpu), &tanh_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::tanh_backward(Tensor grad_output, Tensor output)->Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(tanh_backward_hpu),
                    &tanh_backward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::tanh_(Tensor(a!) self) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(tanh_hpu_), &tanh_hpu_>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(tanh_out_hpu), &tanh_out_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::gelu(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(gelu_hpu), &gelu_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::erf_(Tensor(a!) self) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(erf_hpu_), &erf_hpu_>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::exp_(Tensor(a!) self) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(exp_hpu_), &exp_hpu_>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(neg_out_hpu), &neg_out_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(reciprocal_hpu_), &reciprocal_hpu_>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::reciprocal(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(reciprocal_hpu), &reciprocal_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(reciprocal_out_hpu), &reciprocal_out_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
