#include <torch/script.h>

#include "habana_device/hpu_cached_devices.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h" // TODO: remove after changing allocator
#include "kernel_utils.h"

using namespace torch;


std::vector<synLaunchTensorInfo> generate_syn_launch_tensor_info(
    const std::vector<std::string>& in_names,
    const std::vector<void*>& in_buffers,
    const std::vector<std::string>& out_names,
    const std::vector<void*>& out_buffers) {
  TORCH_CHECK(in_names.size() == in_buffers.size());
  TORCH_CHECK(out_names.size() == out_buffers.size());

  std::vector<synLaunchTensorInfo> syn_info;
  syn_info.reserve(in_names.size() + out_names.size());

  for (size_t i = 0; i < in_names.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfo{
        in_names[i].c_str(), reinterpret_cast<uint64_t>(in_buffers[i])});
  for (size_t i = 0; i < out_names.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfo{
        out_names[i].c_str(), reinterpret_cast<uint64_t>(out_buffers[i])});

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
          &recipe_handle, graph_handle, recipe_name.c_str(), nullptr),
      "synGraphCompile failed");

  { // stream handle scope
    synStreamHandle stream_handle;
    TORCH_HABANA_CHECK(
        synStreamCreate(&stream_handle, device_id, 0),
        "synStreamCreate failed");

    auto syn_launch_info = generate_syn_launch_tensor_info(
        input_names, input_buffers, output_names, output_buffers);

    uint64_t workspace_size_bytes;
    TORCH_HABANA_CHECK(
        synWorkspaceGetSize(&workspace_size_bytes, recipe_handle),
        "synWorkspaceGetSize failed");

    TORCH_HABANA_CHECK(
        synLaunch(
            stream_handle,
            syn_launch_info.data(),
            syn_launch_info.size(),
            synapse_helpers::HPURegistrar::get_device(device_id).get_workspace_buffer(workspace_size_bytes),
            recipe_handle),
        "synLaunch failed");
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");

    TORCH_HABANA_CHECK(
        synStreamDestroy(stream_handle), "synStreamDestroy failed");
  }
}
