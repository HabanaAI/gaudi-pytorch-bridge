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
#include <tpc_kernels/include/perf_lib_layer_params.h>

#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "simple_generic_kernel.h"

using namespace torch;

ns_NLLLossKernel::Params synapse_nll_loss_params_builder(int64_t reduction) {
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
  auto modified_target = std::make_unique<Tensor>();
  if (target.scalar_type() == c10::ScalarType::Long)
    *modified_target =
        target.to("cpu").to(c10::ScalarType::Int).to(target.device());

  synapse_simple_generic_kernel(
      {&output},
      {&self, modified_target->defined() ? &*modified_target : &target},
      "nll_loss",
      &param,
      sizeof(param),
      true);
  LOG_FUNC_END;
  // Note: 2nd output is used in weighted version of this kernel
  return std::make_tuple(output, at::empty({0}, self.options()));
}

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

  auto param = synapse_nll_loss_params_builder(reduction);
  auto grad_input = at::empty(self.sizes(), self.options());
  auto modified_target = std::make_unique<Tensor>();
  if (target.scalar_type() == c10::ScalarType::Long)
    *modified_target =
        target.to("cpu").to(c10::ScalarType::Int).to(target.device());

  synapse_simple_generic_kernel(
      {&grad_input},
      {&grad_output, modified_target->defined() ? &*modified_target : &target},
      "nll_loss",
      &param,
      sizeof(param),
      false);

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
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
