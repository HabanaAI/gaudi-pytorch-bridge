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
  // TORCH_CHECK(!weight.defined(), "weighted nll_loss is not yet supported")
  // TORCH_CHECK(ignore_index == -100, "ignore_index is not yet supported")

  // auto param = synapse_nll_loss_params_builder(reduction);
  // auto output = at::empty({1}, self.options());
  // auto modified_target = std::make_unique<Tensor>();
  // if (target.scalar_type() == c10::ScalarType::Long)
  //   *modified_target =
  //       target.to("cpu").to(c10::ScalarType::Int).to(target.device());

  // synapse_simple_generic_kernel(
  //     {&output},
  //     {&self, modified_target->defined() ? &*modified_target : &target},
  //     "nll_loss",
  //     &param,
  //     sizeof(param),
  //     SynapsePassType::FORWARD_PASS);
  // LOG_FUNC_END;

  // // Note: pytorch expects 0d tensor (scalar)
  // output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  // // Note: 2nd output is used in weighted version of this kernel
  // return std::make_tuple(output, at::empty({0}, self.options()));

  TORCH_WARN("nll_loss_forward_hpu executes CPU kernel internally");
  auto hpu = self.device();
  auto result = at::native::nll_loss_forward_cpu(
      habana_helpers::to_cpu(self),
      habana_helpers::to_cpu(target),
      habana_helpers::to_cpu(weight),
      reduction,
      ignore_index);
  auto ret1 = std::get<0>(result);
  auto ret2 = std::get<1>(result);
  return std::make_tuple(ret1.to(hpu), ret2.to(hpu));
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
  auto grad_input = at::empty(self.sizes(), self.options());

  // auto param = synapse_nll_loss_params_builder(reduction);
  // auto modified_target = std::make_unique<Tensor>();
  // if (target.scalar_type() == c10::ScalarType::Long)
  //   *modified_target =
  //       target.to("cpu").to(c10::ScalarType::Int).to(target.device());

  // synapse_simple_generic_kernel(
  //     {&grad_input},
  //     {&grad_output, modified_target->defined() ? &*modified_target :
  //     &target}, "nll_loss", &param, sizeof(param), SynapsePassType::BACKWARD_PASS);
  // LOG_FUNC_END;
  // return grad_input;

  TORCH_WARN("nll_loss_backward_hpu executes CPU kernel internally");
  auto grad_input_hpu = habana_helpers::to_cpu(grad_input);
  auto hpu = self.device();
  auto cpu_grad_output = habana_helpers::to_cpu(grad_output);
  auto cpu_self = habana_helpers::to_cpu(self);
  auto cpu_target = habana_helpers::to_cpu(target);
  auto cpu_weight = habana_helpers::to_cpu(weight);
  auto cpu_total_weight = habana_helpers::to_cpu(total_weight);

  auto cpu_grad_input =
      habana_helpers::to_cpu(at::empty(self.sizes(), self.options()));
  at::native::nll_loss_backward_out_cpu(
      cpu_grad_input,
      cpu_grad_output,
      cpu_self,
      cpu_target,
      cpu_weight,
      reduction,
      ignore_index,
      cpu_total_weight);

  LOG_FUNC_END;
  return cpu_grad_input.to(hpu);
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
