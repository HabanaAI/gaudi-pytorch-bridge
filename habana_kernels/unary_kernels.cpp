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
    const std::vector<std::string> input_names{"input"};
    const std::vector<std::string> output_names{"output"};
    std::vector<synapse_helpers::tensor> syn_helper_inputs{};
    std::vector<synapse_helpers::tensor> syn_helper_outputs{};
    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(input, input_names[0], true));
    syn_helper_outputs.push_back(
        habana_helpers::create_tensor(output, output_names[0], true));

    std::vector<synTensor> syn_inputs(syn_helper_inputs.size());
    std::vector<synTensor> syn_outputs(syn_helper_outputs.size());
    std::transform(
        syn_helper_inputs.begin(),
        syn_helper_inputs.end(),
        syn_inputs.begin(),
        [](auto& x) { return x.get(); });
    std::transform(
        syn_helper_outputs.begin(),
        syn_helper_outputs.end(),
        syn_outputs.begin(),
        [](auto& x) { return x.get(); });

    {
      const std::string node_type = "relu_fwd_f32";
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
          input_names,
          output_names,
          {input.data_ptr()},
          {output.data_ptr()},
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

Tensor habana_relu(const Tensor& input) {
  std::cout << "habana_relu called\n"; // TODO: remove
  TORCH_CHECK(
      input.scalar_type() == c10::ScalarType::Float,
      "input tensor is not float32");
  auto output = at::empty(input.sizes(), input.options());
  synapse_relu(output, input);

  return output;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema("aten::relu(Tensor self) -> Tensor")
        .impl_unboxedOnlyKernel<decltype(habana_relu), &habana_relu>(
            TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
