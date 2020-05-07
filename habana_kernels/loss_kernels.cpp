/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/core/Reduction.h>
#include <perf_lib_layer_params.h>

#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "simple_generic_kernel.h"

using namespace torch;

static ns_NLLLossKernel::Params synapse_nll_loss_params_builder(
    int64_t reduction) {
  auto param = ns_NLLLossKernel::Params{};
  if (reduction == at::Reduction::Reduction::None) {
    param.mode = NLLLossMode_t::NLL_LOSS_MODE_NONE;
  } else if (reduction == at::Reduction::Reduction::Mean) {
    param.mode = NLLLossMode_t::NLL_LOSS_MODE_MEAN;
  } else if (reduction == at::Reduction::Reduction::Sum) {
    param.mode = NLLLossMode_t::NLL_LOSS_MODE_SUM;
  } else
    TORCH_CHECK(false, "nll_loss got unsuported reduction type: ", reduction);

  return param;
}

static ns_MSELossKernel::Params synapse_mse_loss_params_builder(
    int64_t reduction) {
  auto param = ns_MSELossKernel::Params{};
  if (reduction == at::Reduction::Reduction::None) {
    param.mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_NONE;
  } else if (reduction == at::Reduction::Reduction::Mean) {
    param.mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_MEAN;
  } else if (reduction == at::Reduction::Reduction::Sum) {
    param.mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_SUM;
  } else
    TORCH_CHECK(false, "mse_loss got unsuported reduction type: ", reduction);

  return param;
}

/** @brief Function implements forward pass for torch.nn.NLLLoss
 *  @param self: Input tensor of shape (N,C), where C = Number of classes.
 *  @param target: Input tensor of shape (N), where each value 0 <= i < C.
 *  @param weight: (Tensor, Optional) a manual rescaling weight given to each
 * class. If given, it has to be a Tensor of size C. Otherwise, it is treated as
 * if having all ones.
 *  @param reduction: (String, Optional) Specifies the reduction to apply to the
 * output: 'none' | 'mean' | 'sum'.
 *  @param ignore_index: (Long, Optional) Specifies a target value that is
 * ignored and does not contribute to the input gradient.
 */
std::tuple<Tensor, Tensor> nll_loss_forward_hpu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  LOG_FUNC_BEGIN;
  TORCH_CHECK(!weight.defined(), "weighted nll_loss is not yet supported")
  TORCH_CHECK(ignore_index == -100, "ignore_index is not yet supported")

  auto param = synapse_nll_loss_params_builder(reduction);
  auto output = at::empty({1}, self.options());
  auto modified_target = habana_helpers::cast_tensor_to_integer(target);

  synapse_simple_generic_kernel(
      {&output},
      {&self, &modified_target},
      "nll_loss",
      &param,
      sizeof(param),
      SynapsePassType::FORWARD_PASS);
  LOG_FUNC_END;

  // Note: pytorch expects 0d tensor (scalar)
  output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  // Note: 2nd output is used in weighted version of this kernel
  return std::make_tuple(output, at::empty({0}, self.options()));
}

/** @brief Function implements backward pass for torch.nn.NLLLoss
 *  @param grad_output: Input (bwd_pass) tensor of shape N or 1.
 *  @param self: Input (fwd_pass) tensor of shape (N,C), where C = Number of
 * classes.
 *  @param target: Input tensor (fwd_pass) of shape (N), where each value 0 <= i
 * < C.
 *  @param weight: (Tensor, Optional) a manual rescaling weight given to each
 * class. If given, it has to be a Tensor of size C. Otherwise, it is treated as
 * if having all ones.
 *  @param reduction: (String, Optional) Specifies the reduction to apply to the
 * output: 'none' | 'mean' | 'sum'.
 *  @param ignore_index: (Long, Optional) Specifies a target value that is
 * ignored and does not contribute to the input gradient.
 *  @param total_weight: (single element tensor) sum of weights used in fwd_pass
 */
Tensor nll_loss_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    UNUSED const Tensor& total_weight) {
  LOG_FUNC_BEGIN;
  TORCH_CHECK(!weight.defined(), "weighted nll_loss is not yet supported")
  TORCH_CHECK(ignore_index == -100, "ignore_index is not yet supported")

  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (grad_output.dim() == 0) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto grad_input = at::empty(self.sizes(), self.options());

  auto param = synapse_nll_loss_params_builder(reduction);
  auto modified_target = habana_helpers::cast_tensor_to_integer(target);

  synapse_simple_generic_kernel(
      {&grad_input},
      {&grad_output, &modified_target},
      "nll_loss",
      &param,
      sizeof(param),
      SynapsePassType::BACKWARD_PASS);
  LOG_FUNC_END;

  return grad_input;
}

/** @brief Function implements forward pass for torch.nn.MSELoss
 *  @param self: Input tensor of shape (N,C)
 *  @param target: Input tensor of shape (N,C)
 *  @param reduction: (String, Optional) Specifies the reduction to apply to the
 * output: 'none' | 'mean' | 'sum'.
 */
Tensor mse_loss_forward_hpu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  LOG_FUNC_BEGIN;

  auto param = synapse_mse_loss_params_builder(reduction);

  Tensor output;
  if (reduction == at::Reduction::Reduction::None) {
    output = at::empty(self.sizes(), self.options());
  } else {
    output = at::empty({1}, self.options());
  }

  synapse_simple_generic_kernel(
      {&output},
      {&self, &target},
      "mse_loss",
      &param,
      sizeof(param),
      SynapsePassType::FORWARD_PASS);
  LOG_FUNC_END;

  if (reduction != at::Reduction::Reduction::None) {
    // Note: pytorch expects 0d tensor (scalar)
    output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  return output;
}

/** @brief Function implements backward pass for torch.nn.MSELoss
 *  @param grad_output: Input (bwd_pass) tensor of shape (N,C) or 1.
 *  @param self: Input (fwd_pass) tensor of shape (N,C)
 *  @param target: Input tensor (fwd_pass) of shape (N,C)
 *  @param reduction: (String, Optional) Specifies the reduction to apply to the
 * output: 'none' | 'mean' | 'sum'.
 */
Tensor mse_loss_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  LOG_FUNC_BEGIN;

  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (grad_output.dim() == 0) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto grad_input = at::empty(self.sizes(), self.options());

  auto param = synapse_mse_loss_params_builder(reduction);

  synapse_simple_generic_kernel(
      {&grad_input},
      {&grad_output, &self, &target},
      "mse_loss",
      &param,
      sizeof(param),
      SynapsePassType::BACKWARD_PASS);
  LOG_FUNC_END;
  return grad_input;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) ->(Tensor output, Tensor total_weight)")
                .impl_unboxedOnlyKernel<
                    decltype(nll_loss_forward_hpu),
                    &nll_loss_forward_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(nll_loss_backward_hpu),
                    &nll_loss_backward_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(mse_loss_forward_hpu),
                    &mse_loss_forward_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(mse_loss_backward_hpu),
                    &mse_loss_backward_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));