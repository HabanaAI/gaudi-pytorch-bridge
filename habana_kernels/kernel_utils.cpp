#include <torch/script.h>
#include <tuple>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h" // TODO: remove after changing allocator
#include "kernel_utils.h"

using namespace torch;

synTensorDescriptorTr habana_helpers::synapse_tensor_descriptor_builder(
    const c10::IntArrayRef& shape,
    const synDataType dtype,
    const std::string& name,
    const bool persistent) {
  synTensorDescriptorTr descriptor;
  descriptor.m_dataType = dtype;
  descriptor.m_dims = shape.size();

  TORCH_CHECK(
      shape.size() <= SYN_MAX_TENSOR_DIM,
      name,
      " tensor has more than ",
      SYN_MAX_TENSOR_DIM,
      " dimensions");
  // write NHWC as CWHN and write 0 at the end
  for (int i = 0; i < shape.size(); ++i)
    descriptor.m_sizes[i] = shape[shape.size() - i - 1];
  for (int i = shape.size(); i < SYN_MAX_TENSOR_DIM; ++i)
    descriptor.m_sizes[i] = 0;

  //   descriptor.m_strides[SYN_MAX_TENSOR_DIM]; // TODO: not needed?
  descriptor.m_name = name.c_str(); // TODO: we take only pointer, so make sure
                                    // name object will be alive
  descriptor.m_deviceMemAddress = 0; // It will be patched during runtime
  descriptor.m_isOutput = persistent;
  descriptor.m_isPersistent = persistent;

  return descriptor;
}

std::tuple<at::DataPtr, at::DataPtr> allocate_workspace_and_topology_buffers(
    const synRecipeHandle recipe_handle,
    at::Allocator* allocator) {
  uint64_t topology_size_bytes, workspace_size_bytes;
  TORCH_HABANA_CHECK(
      synRecipeGetSize(&topology_size_bytes, recipe_handle),
      "synRecipeGetSize failed");
  TORCH_HABANA_CHECK(
      synWorkspaceGetSize(&workspace_size_bytes, recipe_handle),
      "synWorkspaceGetSize failed");

  at::DataPtr topology_buffer = allocator->allocate(topology_size_bytes);
  at::DataPtr workspace_buffer;
  if (workspace_size_bytes)
    workspace_buffer = allocator->allocate(workspace_size_bytes);

  return {std::move(topology_buffer), std::move(workspace_buffer)};
}

std::vector<synLaunchTensorInfo> generate_syn_launch_tensor_info(
    const std::vector<std::string>& names,
    const std::vector<void*>& buffers) {
  TORCH_CHECK(names.size() == buffers.size());

  std::vector<synLaunchTensorInfo> syn_info;
  syn_info.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfo{
        names[i].c_str(), reinterpret_cast<uint64_t>(buffers[i])});

  return syn_info;
}

std::string habana_helpers::unique_recipe_name_generator(
    std::string recipe_name) {
  static std::unordered_map<std::string, unsigned> map;
  return recipe_name + std::to_string(map[recipe_name]++);
}

void habana_helpers::compile_and_run(
    const std::string& recipe_prefix,
    const synGraphHandle graph_handle,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    const uint32_t device_id) {
  synRecipeHandle recipe_handle;
  const auto recipe_name =
      habana_helpers::unique_recipe_name_generator(recipe_prefix);
  TORCH_HABANA_CHECK(
      synGraphCompile(
          &recipe_handle, graph_handle, recipe_name.c_str(), nullptr, 0, 0),
      "synGraphCompile failed");

  at::DataPtr topology_buffer, workspace_buffer;
  std::tie(topology_buffer, workspace_buffer) =
      allocate_workspace_and_topology_buffers(
          recipe_handle,
          at::habana::getHABANADeviceAllocator()); // TODO: use different
                                                   // allocator

  { // recipe upload scope
    const synRecipeInfo recipe_info{
        recipe_name.c_str(), reinterpret_cast<uint64_t>(topology_buffer.get())};
    TORCH_HABANA_CHECK(
        synRecipeUpload(recipe_handle, &recipe_info, device_id),
        "synRecipeUpload failed");
    { // stream handle scope
      synStreamHandle stream_handle;
      TORCH_HABANA_CHECK(
          synStreamCreate(&stream_handle, device_id, 0),
          "synStreamCreate failed");

      auto syn_inputs_info =
          generate_syn_launch_tensor_info(input_names, input_buffers);
      auto syn_outputs_info =
          generate_syn_launch_tensor_info(output_names, output_buffers);

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
          synStreamSynchronize(stream_handle), "synStreamSynchronize failed");

      TORCH_HABANA_CHECK(
          synStreamDestroy(stream_handle), "synStreamDestroy failed");
    }
    TORCH_HABANA_CHECK(
        synRecipeUnload(recipe_handle, &recipe_info, device_id),
        "synRecipeUnload failed");
  }
}
