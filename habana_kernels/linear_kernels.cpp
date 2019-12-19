#include <ATen/InferSize.h>
#include <torch/script.h>
#include <tpc_kernels/include/perf_lib_layer_params.h>
#include <algorithm>
#include <iostream>

#include "conv_pool_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_helpers/tensor_utils.h"
#include "kernel_utils.h"

using namespace torch;

void synapse_matmul(
    const Tensor& output,
    const Tensor& mat1,
    const Tensor& mat2) {
  const auto device_id = mat1.device().index();
  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");
  { // tensors scope
    const std::vector<std::string> input_names{"mat1", "mat2"};
    const std::vector<std::string> output_names{"output"};

    std::vector<synapse_helpers::tensor> syn_helper_inputs{};
    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(mat1, input_names[0], true));
    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(mat2, input_names[1], true));
    std::vector<synapse_helpers::tensor> syn_helper_outputs{};
    syn_helper_outputs.push_back(
        habana_helpers::create_tensor(output, output_names[0], true));
    // workaround for missing synapse_helpers::graph support
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


      const std::string node_type = "gemm";
      { // add node
        synGEMMParams params{false, false};

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
      }

      habana_helpers::compile_and_run(
          node_type,
          graph_handle,
          input_names,
          output_names,
          {mat1.data_ptr(), mat2.data_ptr()},
          {output.data_ptr()},
          device_id);
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

Tensor habana_matmul(const Tensor& mat1, const Tensor& mat2) {
  std::cout << "habana_matmul called\n"; // TODO: remove

  TORCH_CHECK(
      mat1.ndimension() == 2, "habana_matmul supports only 2d matrices");
  TORCH_CHECK(
      mat2.ndimension() == 2, "habana_matmul supports only 2d matrices");
  TORCH_CHECK(
      mat1.size(1) == mat2.size(0), "matmul inner dimensions doesn't match");

  auto output = at::empty({mat1.size(0), mat2.size(1)}, mat1.options());
  synapse_matmul(output, mat1, mat2);

  return output;
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema("aten::mm(Tensor self, Tensor mat2) -> Tensor")
        .impl_unboxedOnlyKernel<decltype(habana_matmul), &habana_matmul>(
            TensorTypeId::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
