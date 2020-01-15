#include <torch/script.h>
#include <tpc_kernels/include/perf_lib_layer_params.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

void synapse_log_softmax(
    const Tensor& output,
    const Tensor& input,
    const int64_t dim) {
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

    const std::vector<std::string> temp_names{"tmp"};
    std::vector<synapse_helpers::tensor> syn_helper_temps{};
    // temp tensor will have the same properties as output
    syn_helper_temps.push_back(habana_helpers::create_tensor(
        output, temp_names[0], graph_handle, false));

    std::vector<synTensor> syn_temps(syn_helper_temps.size());
    std::transform(
        syn_helper_temps.begin(),
        syn_helper_temps.end(),
        syn_temps.begin(),
        [](auto& x) { return x.get(); });
    {
      // TODO: use logsoftmax_fwd_f32 instead of log + softmax
      const std::string node1_type = "log_fwd_" +
          habana_helpers::name_suffix_from_type(input.scalar_type());
      const std::string node2_type = "softmax_fwd_" +
          habana_helpers::name_suffix_from_type(input.scalar_type());

      TORCH_CHECK(
          dim == 1, "Trying to run softmax with dim other than channels");
      ns_Softmax::Params params{static_cast<int>(dim)};
      { // add node
        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_inputs.data(),
                syn_temps.data(),
                syn_inputs.size(),
                syn_temps.size(),
                nullptr,
                0,
                node1_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");

        TORCH_HABANA_CHECK(
            synNodeCreate(
                graph_handle,
                syn_temps.data(),
                syn_outputs.data(),
                syn_temps.size(),
                syn_outputs.size(),
                &params,
                sizeof(params),
                node2_type.c_str(),
                "",
                nullptr,
                nullptr),
            "synNodeCreate failed");
      }

      habana_helpers::compile_and_run(
          node2_type + node1_type,
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

Tensor log_softmax_hpu(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  std::cout << "log_softmax_hpu called\n"; // TODO: remove

  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");

  // TODO: consider changing layout like it is done in convolution
  auto output = at::empty(self.sizes(), self.options());
  synapse_log_softmax(output, self, dim);

  return output;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
        .impl_unboxedOnlyKernel<decltype(log_softmax_hpu), &log_softmax_hpu>(
            TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
