/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;

/** log_softmax (forward pass) implementation for Habana device
 * @params [In] self: Input tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] half_to_float:
 */
Tensor log_softmax_hpu(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");

  auto output = at::empty(self.sizes(), self.options());
  ns_Softmax::Params params{static_cast<int>(self.ndimension() - 1 - dim)};

  std::vector<const at::Tensor*> pt_inputs{&self};
  std::vector<const at::Tensor*> pt_outputs{&output};
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "logsoftmax",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return output;
}

/** log_softmax (backward pass) implementation for Habana device
 * @params [In] grad: Backward pass Input tensor. 2-4D. bf16, fp32
 * @params [In] output: Forward pass Output tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] input: Forward pass Input tensor. 2-4D. bf16, fp32
 */
Tensor log_softmax_backward_hpu(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  LOG_FUNC_BEGIN;

  auto input_grad = at::empty(input.sizes(), input.options());
  ns_Softmax::Params params{static_cast<int>(input.ndimension() - 1 - dim)};

  std::vector<const at::Tensor*> pt_inputs{&output, &grad};
  std::vector<const at::Tensor*> pt_outputs{&input_grad};
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "logsoftmax",
      &params,
      sizeof(params),
      SynapsePassType::BACKWARD_PASS);

  LOG_FUNC_END;
  return input_grad;
}

/** softmax (forward pass) implementation for Habana device
 * @params [In] self: Input tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] half_to_float:
 */
Tensor softmax_hpu(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");

  auto output = at::empty(self.sizes(), self.options());
  ns_Softmax::Params params{static_cast<int>(self.ndimension() - 1 - dim)};

  std::vector<const at::Tensor*> pt_inputs{&self};
  std::vector<const at::Tensor*> pt_outputs{&output};
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "softmax",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return output;
}

/** softmax (backward pass) implementation for Habana device
 * @params [In] grad: Backward pass Input tensor. 2-4D. bf16, fp32
 * @params [In] output: Forward pass Output tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] input: Forward pass Input tensor. 2-4D. bf16, fp32
 */
Tensor softmax_backward_hpu(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  LOG_FUNC_BEGIN;

  auto input_grad = at::empty(input.sizes(), input.options());
  ns_Softmax::Params params{static_cast<int>(input.ndimension() - 1 - dim)};

  std::vector<const at::Tensor*> pt_inputs{&output, &grad};
  std::vector<const at::Tensor*> pt_outputs{&input_grad};
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "softmax",
      &params,
      sizeof(params),
      SynapsePassType::BACKWARD_PASS);

  LOG_FUNC_END;
  return input_grad;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(log_softmax_hpu),
                    &log_softmax_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(log_softmax_backward_hpu),
                    &log_softmax_backward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(softmax_hpu),
                    &softmax_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(softmax_backward_hpu),
                    &softmax_backward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
