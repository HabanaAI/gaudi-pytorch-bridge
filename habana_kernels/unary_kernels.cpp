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

  synapse_simple_generic_inplace_kernel(pt_inputs, "relu", nullptr, 0, SynapsePassType::FORWARD_PASS);

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
      pt_outputs, pt_inputs, "sigmoid", nullptr, 0, SynapsePassType::FORWARD_PASS);

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
      pt_outputs, pt_inputs, "sigmoid", nullptr, 0, SynapsePassType::BACKWARD_PASS);

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
static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema("aten::relu_(Tensor(a!) self) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(relu_hpu_), &relu_hpu_>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::relu(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(relu_hpu), &relu_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::sigmoid(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(sigmoid_hpu), &sigmoid_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(sigmoid_backward_hpu), &sigmoid_backward_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::sqrt(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(sqrt_hpu), &sqrt_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
