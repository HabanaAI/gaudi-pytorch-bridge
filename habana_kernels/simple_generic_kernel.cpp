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
#include "habana_helpers/graph.h"

void synapse_simple_generic_kernel(
    std::vector<const at::Tensor*> pt_outputs, // NHWC
    std::vector<const at::Tensor*> pt_inputs, // NHWC
    const std::string& node_guid,
    void* syn_param,
    const size_t syn_param_size,
    const SynapsePassType pass_type) {
  size_t device_id;
  at::ScalarType scalar_type;
  // RNG kernels have 0 inputs and 1 output
  if (pt_inputs.size()) {
    device_id = pt_inputs[0]->device().index();
    scalar_type = pt_inputs[0]->scalar_type();
  } else {
    device_id = pt_outputs[0]->device().index();
    scalar_type = pt_outputs[0]->scalar_type();
  }
  std::string node_type = (SynapsePassType::NO_PASS == pass_type) ? node_guid
                                                                  : node_guid +
          std::string((SynapsePassType::FORWARD_PASS == pass_type) ? "_fwd_"
                                                                   : "_bwd_") +
          habana_helpers::name_suffix_from_type(scalar_type);
  auto graph = habana_helpers::create_graph(device_id, node_type);
  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs;
    std::vector<synTensor> syn_inputs, syn_outputs;

    std::tie(syn_helper_inputs, syn_inputs) = habana_helpers::create_tensors(
        pt_inputs, graph.get_graph_handle(), true);
    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        pt_outputs, graph.get_graph_handle(), true);
    {
      graph.add_node(
          std::move(syn_inputs),
          std::move(syn_outputs),
          syn_param,
          syn_param_size,
          std::move(node_type));

      habana_helpers::compile_and_run(
          std::move(graph),
          habana_helpers::names(syn_helper_inputs),
          habana_helpers::names(syn_helper_outputs),
          habana_helpers::extract_data_ptrs(pt_inputs),
          habana_helpers::extract_data_ptrs(pt_outputs),
          device_id);
    }
  }
}

void synapse_simple_generic_inplace_kernel(
    std::vector<const at::Tensor*> pt_inputs, // NHWC
    const std::string& node_guid,
    void* syn_param,
    const size_t syn_param_size,
    const SynapsePassType pass_type) {
  const auto device_id = pt_inputs[0]->device().index();
  std::string node_type = (SynapsePassType::NO_PASS == pass_type) ? node_guid
                                                                  : node_guid +
          std::string((SynapsePassType::FORWARD_PASS == pass_type) ? "_fwd_"
                                                                   : "_bwd_") +
          habana_helpers::name_suffix_from_type(pt_inputs[0]->scalar_type());
  auto graph = habana_helpers::create_graph(device_id, node_type);
  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs;
    std::vector<synTensor> syn_inputs, syn_outputs;

    std::tie(syn_helper_inputs, syn_inputs) = habana_helpers::create_tensors(
        pt_inputs, graph.get_graph_handle(), true);
    auto syn_helper_output = habana_helpers::duplicate_tensor_in_memory_section(
        syn_helper_inputs[0]);
    syn_outputs.push_back(syn_helper_output.get());
    {
      graph.add_node(
          std::move(syn_inputs),
          std::move(syn_outputs),
          syn_param,
          syn_param_size,
          std::move(node_type));

      habana_helpers::compile_and_run(
          std::move(graph),
          habana_helpers::names(syn_helper_inputs),
          {syn_helper_output.tensor_name_},
          habana_helpers::extract_data_ptrs(pt_inputs),
          {pt_inputs[0]->data_ptr()},
          device_id);
    }
  }
}
