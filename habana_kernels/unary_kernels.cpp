#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

void synapse_relu(const Tensor& output, const Tensor& input) {
  auto& device =
      synapse_helpers::HPURegistrar::get_device(input.device().index());
  const auto device_id = device.id();

  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");

  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs;
    std::vector<synTensor> syn_inputs, syn_outputs;

    std::tie(syn_helper_inputs, syn_inputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&input},
        {"input"},
        graph_handle,
        {true});
    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output},
        {"output"},
        graph_handle,
        {true});

    {
      const std::string node_type = "relu_fwd_" +
          habana_helpers::name_suffix_from_type(input.scalar_type());
      { // add relu node
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_inputs.data(),
                syn_outputs.data(),
                syn_inputs.size(),
                syn_outputs.size(),
                nullptr,
                0,
                node_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }

      habana_helpers::compile_and_run(
          node_type,
          graph_handle,
          habana_helpers::names(syn_helper_inputs),
          habana_helpers::names(syn_helper_outputs),
          {input.data_ptr()},
          {output.data_ptr()},
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

Tensor relu_hpu(const Tensor& input) {
  std::cout << "relu_hpu called\n"; // TODO: remove
  auto output = at::empty(input.sizes(), input.options());
  synapse_relu(output, input);

  return output;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema("aten::relu(Tensor self) -> Tensor")
        .impl_unboxedOnlyKernel<decltype(relu_hpu), &relu_hpu>(
            TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
