#include <ATen/ExpandUtils.h>
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

void check_matmul_params(const Tensor& mat1, const Tensor& mat2) {
  TORCH_CHECK(mat1.ndimension() == 2, "matmul_hpu supports only 2d matrices");
  TORCH_CHECK(mat2.ndimension() == 2, "matmul_hpu supports only 2d matrices");
  TORCH_CHECK(
      mat1.size(1) == mat2.size(0), "matmul inner dimensions doesn't match");
}

// TODO: mat2 transposed or not?
// output = mat1 x mat2
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
        std::vector<const at::Tensor*>{&mat1, &mat2}, graph_handle, true);
    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output}, graph_handle, true);

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

// TODO: mat2 transposed or not?
// output = alpha * mat1 x mat2 + beta * bias
void synapse_matmul(
    const Tensor& output,
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& bias,
    const Scalar& beta,
    const Scalar& alpha) {
  // TODO: implement support for scalars
  TORCH_CHECK(
      beta.to<int>() == 1, "matmul_with_bias_hpu doesn't support scalars yet");
  TORCH_CHECK(
      alpha.to<int>() == 1, "matmul_with_bias_hpu doesn't support scalars yet");
  const auto device_id = mat1.device().index();
  // graph_handle scope
  synGraphHandle graph_handle;
  TORCH_HABANA_CHECK(
      synGraphCreate(&graph_handle, synDeviceType::synDeviceGaudi),
      "synGraphCreate failed");
  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs,
        syn_tmp_helper_tensors;
    std::vector<synTensor> syn_inputs, syn_outputs, syn_tmp_tensors;

    std::tie(syn_helper_inputs, syn_inputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&mat1, &mat2, &bias},
        graph_handle,
        true);
    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output}, graph_handle, true);
    std::tie(syn_tmp_helper_tensors, syn_tmp_tensors) =
        habana_helpers::create_tensors(
            std::vector<const at::Tensor*>{&output}, graph_handle, false);

    const std::string node_type1 = "gemm";
    const std::string node_type2 =
        "add_fwd_" + habana_helpers::name_suffix_from_type(mat1.scalar_type());
    { // add node
      synGEMMParams params{false, false};

      TORCH_HABANA_CHECK(
          synNodeCreate(
              graph_handle,
              syn_inputs.data(),
              syn_tmp_tensors.data(),
              2, // syn_inputs has bias add at the end
              syn_tmp_tensors.size(),
              &params,
              sizeof(params),
              node_type1.c_str(),
              "",
              nullptr,
              nullptr),
          "synNodeCreate failed");
    }
    { // add node
      syn_tmp_tensors.push_back(syn_inputs[2]); // mm_out + bias
      TORCH_HABANA_CHECK(
          synNodeCreate(
              graph_handle,
              syn_tmp_tensors.data(),
              syn_outputs.data(),
              syn_tmp_tensors.size(),
              syn_outputs.size(),
              nullptr,
              0,
              node_type2.c_str(),
              "",
              nullptr,
              nullptr),
          "synNodeCreate failed");
    }
    habana_helpers::compile_and_run(
        node_type1 + node_type2,
        graph_handle,
        habana_helpers::names(syn_helper_inputs),
        habana_helpers::names(syn_helper_outputs),
        {mat1.data_ptr(), mat2.data_ptr(), bias.data_ptr()},
        {output.data_ptr()},
        device_id);
  }
  TORCH_HABANA_CHECK(synGraphDestroy(graph_handle), "synGraphDestroy failed");
}

Tensor matmul_hpu(const Tensor& mat1, const Tensor& mat2) {
  LOG_FUNC_BEGIN;
  check_matmul_params(mat1, mat2);

  auto output = at::empty({mat1.size(0), mat2.size(1)}, mat1.options());
  synapse_matmul(output, mat1, mat2);
  LOG_FUNC_END;
  return output;
}

Tensor matmul_with_bias_hpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  LOG_FUNC_BEGIN;
  check_matmul_params(mat1, mat2);
  TORCH_CHECK(
      self.sizes().size() == 1,
      "Bias must be 1D tensor, but it has ",
      self.sizes().size(),
      " dimensions.");
  TORCH_CHECK(
      self.size(0) == mat2.size(1),
      "Sizes don't match, ",
      self.size(0),
      " vs ",
      mat2.size(1));

  auto output = at::empty({mat1.size(0), mat2.size(1)}, mat1.options());
  Tensor bias_expanded;
  std::tie(bias_expanded) =
      at::expand_size(self, output.sizes(), "matmul_with_bias_hpu");
  synapse_matmul(output, mat1, mat2, bias_expanded, beta, alpha);
  LOG_FUNC_END;
  return output;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema("aten::mm(Tensor self, Tensor mat2) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(matmul_hpu), &matmul_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta = 1, Scalar alpha = 1) ->Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(matmul_with_bias_hpu),
                    &matmul_with_bias_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
