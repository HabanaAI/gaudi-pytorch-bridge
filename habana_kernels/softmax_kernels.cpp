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
#include <tpc_kernels/include/perf_lib_layer_params.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

void synapse_log_softmax_generic_impl(
    std::vector<const at::Tensor*> pt_outputs,
    std::vector<const at::Tensor*> pt_inputs,
    const int64_t dim,
    const bool forward_pass) {
  auto& device =
      synapse_helpers::HPURegistrar::get_device(pt_inputs[0]->device().index());
  const auto device_id = device.id();

  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");

  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs;
    std::vector<synTensor> syn_inputs, syn_outputs;

    std::tie(syn_helper_inputs, syn_inputs) =
        habana_helpers::create_tensors(pt_inputs, graph_handle, true);
    std::tie(syn_helper_outputs, syn_outputs) =
        habana_helpers::create_tensors(pt_outputs, graph_handle, true);
    {
      const std::string node_type = std::string("logsoftmax_") +
          (forward_pass ? "fwd_" : "bwd_") +
          habana_helpers::name_suffix_from_type(pt_inputs[0]->scalar_type());

      ns_Softmax::Params params{
          static_cast<int>(pt_inputs[0]->ndimension() - 1 - dim)};

      TORCH_HABANA_CHECK(
          synNodeCreate(
              graph_handle,
              syn_inputs.data(),
              syn_outputs.data(),
              syn_inputs.size(),
              syn_outputs.size(),
              &params,
              sizeof(params),
              node_type.c_str(),
              "",
              nullptr,
              nullptr),
          "synNodeCreate failed");

      habana_helpers::compile_and_run(
          node_type,
          graph_handle,
          habana_helpers::names(syn_helper_inputs),
          habana_helpers::names(syn_helper_outputs),
          habana_helpers::extract_data_ptrs(pt_inputs),
          habana_helpers::extract_data_ptrs(pt_outputs),
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

Tensor log_softmax_hpu(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");

  auto output = at::empty(self.sizes(), self.options());
  synapse_log_softmax_generic_impl({&output}, {&self}, dim, true);

  LOG_FUNC_END;
  return output;
}

Tensor log_softmax_backward_hpu(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    UNUSED const Tensor& input) {
  LOG_FUNC_BEGIN;

  auto input_grad = at::empty(grad.sizes(), grad.options());
  synapse_log_softmax_generic_impl({&input_grad}, {&output, &grad}, dim, false);

  TORCH_WARN("log_softmax_backward_hpu executes CPU kernel internally");
  auto hpu = grad.device();
  auto result = at::native::log_softmax_backward_cpu(
      habana_helpers::to_cpu(grad),
      habana_helpers::to_cpu(output),
      dim,
      habana_helpers::to_cpu(input));

  LOG_FUNC_END;
  return result.to(hpu);
  //   return input_grad;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(log_softmax_hpu),
                    &log_softmax_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(log_softmax_backward_hpu),
                    &log_softmax_backward_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
