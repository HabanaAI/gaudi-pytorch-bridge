/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "simple_generic_kernel.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"

void synapse_simple_generic_kernel(
    std::vector<const at::Tensor*> pt_outputs, // NHWC
    std::vector<const at::Tensor*> pt_inputs, // NHWC
    const std::string& node_guid,
    const void* syn_param,
    const size_t syn_param_size,
    const bool forward_pass) {
  const auto device_id = pt_inputs[0]->device().index();
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
      const std::string node_type = node_guid +
          std::string(forward_pass ? "_fwd_" : "_bwd_") +
          habana_helpers::name_suffix_from_type(pt_inputs[0]->scalar_type());
      { // add node
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_inputs.data(),
                syn_outputs.data(),
                syn_inputs.size(),
                syn_outputs.size(),
                syn_param,
                syn_param_size,
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
          habana_helpers::extract_data_ptrs(pt_inputs),
          habana_helpers::extract_data_ptrs(pt_outputs),
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

void synapse_simple_generic_inplace_kernel(
    std::vector<const at::Tensor*> pt_inputs, // NHWC
    const std::string& node_guid,
    const void* syn_param,
    const size_t syn_param_size,
    const bool forward_pass) {
  const auto device_id = pt_inputs[0]->device().index();
  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");
  { // tensors scope

    std::vector<synapse_helpers::tensor> syn_helper_inputs;
    std::vector<synTensor> syn_inputs, syn_outputs;

    std::tie(syn_helper_inputs, syn_inputs) =
        habana_helpers::create_tensors(pt_inputs, graph_handle, true);
    auto syn_helper_output = habana_helpers::duplicate_tensor_in_memory_section(
        syn_helper_inputs[0]);

    {
      const std::string node_type = node_guid +
          std::string(forward_pass ? "_fwd_" : "_bwd_") +
          habana_helpers::name_suffix_from_type(pt_inputs[0]->scalar_type());
      { // add node
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_inputs.data(),
                &syn_helper_output.get(),
                syn_inputs.size(),
                1,
                syn_param,
                syn_param_size,
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
          {syn_helper_output.tensor_name_},
          habana_helpers::extract_data_ptrs(pt_inputs),
          {pt_inputs[0]->data_ptr()},
          device_id);
    }
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}