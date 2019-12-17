#include <synapse/include/synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/fake_tensor_builder.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
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
    syn_helper_inputs.push_back(synapse_helpers::tensor_builder::create_tensor(
        device_id,
        synDataType::syn_type_float,
        input.nbytes(),
        input.sizes().size(),
        input.sizes(),
        input_names[0],
        true));
    syn_helper_outputs.push_back(synapse_helpers::tensor_builder::create_tensor(
        device_id,
        synDataType::syn_type_float,
        output.nbytes(),
        output.sizes().size(),
        output.sizes(),
        output_names[0],
        true));

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

      { // graph compilation, workspace buffer and topology buffer allocation
        synRecipeHandle recipe_handle;
        const auto recipe_name =
            habana_helpers::unique_recipe_name_generator(node_type);
        TORCH_HABANA_CHECK(
            synGraphCompile(
                &recipe_handle,
                graph_handle,
                recipe_name.c_str(),
                nullptr,
                0,
                0),
            "synGraphCompile failed");

        uint64_t topology_size_bytes, workspace_size_bytes;
        TORCH_HABANA_CHECK(
            synRecipeGetSize(&topology_size_bytes, recipe_handle),
            "synRecipeGetSize failed");
        TORCH_HABANA_CHECK(
            synWorkspaceGetSize(&workspace_size_bytes, recipe_handle),
            "synWorkspaceGetSize failed");

        auto hpu_raii_allocator = at::habana::getHABANADeviceAllocator();

        at::DataPtr topology_buffer =
            hpu_raii_allocator->allocate(topology_size_bytes);
        at::DataPtr workspace_buffer;
        if (workspace_size_bytes)
          workspace_buffer = hpu_raii_allocator->allocate(workspace_size_bytes);

        { // recipe upload scope
          const synRecipeInfo recipe_info{
              recipe_name.c_str(),
              reinterpret_cast<uint64_t>(topology_buffer.get())};
          TORCH_HABANA_CHECK(
              synRecipeUpload(recipe_handle, &recipe_info, device_id),
              "synRecipeUpload failed");
          { // stream handle scope
            synStreamHandle stream_handle;
            TORCH_HABANA_CHECK(
                synStreamCreate(&stream_handle, device_id, 0),
                "synStreamCreate failed");

            std::vector<synLaunchTensorInfo> syn_inputs_info{
                {input_names[0].c_str(),
                 reinterpret_cast<uint64_t>(input.data_ptr())},
            };
            std::vector<synLaunchTensorInfo> syn_outputs_info{
                {output_names[0].c_str(),
                 reinterpret_cast<uint64_t>(output.data_ptr())}};

            TORCH_HABANA_CHECK(
                synLaunch(
                    stream_handle,
                    syn_inputs_info.data(),
                    syn_inputs_info.size(),
                    syn_outputs_info.data(),
                    syn_outputs_info.size(),
                    reinterpret_cast<uint64_t>(workspace_buffer.get()),
                    &recipe_info),
                "synLaunch failed");
            TORCH_HABANA_CHECK(
                synStreamSynchronize(stream_handle),
                "synStreamSynchronize failed");

            TORCH_HABANA_CHECK(
                synStreamDestroy(stream_handle), "synStreamDestroy failed");
          }
          TORCH_HABANA_CHECK(
              synRecipeUnload(recipe_handle, &recipe_info, device_id),
              "synRecipeUnload failed");
        }
      }
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
