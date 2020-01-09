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
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs;
    std::vector<synTensor> syn_inputs, syn_outputs;

    std::tie(syn_helper_inputs, syn_inputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&mat1, &mat2},
        {"mat1", "mat2"},
        {true, true});
    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output}, {"output"}, {true});

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
        habana_helpers::names(syn_helper_inputs),
        habana_helpers::names(syn_helper_outputs),
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
