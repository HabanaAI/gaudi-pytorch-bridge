/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <torch/script.h>
#include <perf_lib_layer_params.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_device/tensor_builder.h"
#include "habana_helpers/tensor_utils.h"
#include "kernel_utils.h"

using namespace torch;

// TODO: remove this function. Workaround for SW-9962
[[deprecated]] void adjust_output_tensor_(Tensor& tensor) {
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      {tensor.size(1), tensor.size(0)}, {1, tensor.size(1)});
  tensor = habana_helpers::contiguous_tensor(tensor);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      {tensor.size(1), tensor.size(0)}, {tensor.size(0), 1});
};

void check_matmul_params(
    const Tensor& mat1,
    const Tensor& mat2,
    c10::optional<const at::Tensor*> bias) {
  TORCH_CHECK(mat1.ndimension() == 2, "matmul_hpu supports only 2d matrices");
  TORCH_CHECK(mat2.ndimension() == 2, "matmul_hpu supports only 2d matrices");
  TORCH_CHECK(
      mat1.size(1) == mat2.size(0), "matmul inner dimensions doesn't match");
  TORCH_CHECK(
      static_cast<int>(mat1.is_contiguous()) + mat2.is_contiguous() > 0,
      "Only one matrix can me non contiguous.",
      "\nmat1.is_contiguous() returned: ",
      mat1.is_contiguous(),
      "\nmat2.is_contiguous() returned: ",
      mat2.is_contiguous(),
      "\nmat1 sizes: ",
      mat1.sizes(),
      "mat1 strides: ",
      mat1.strides(),
      "\nmat2 sizes: ",
      mat2.sizes(),
      "mat2 strides: ",
      mat2.strides());
  if (bias)
    TORCH_CHECK(
        bias.value()->ndimension() == 1, "matmul_hpu supports only 1d bias");
}

synapse_helpers::tensor create_hacked_matmul_tensor(
    const Tensor& mat2,
    const synGraphHandle graph_handle,
    const unsigned device_id) {
  if (mat2.is_contiguous())
    return habana_helpers::create_tensor(
        mat2, graph_handle, true, c10::nullopt);
  else {
    // Matrix is 2d and not contiguous, it means tensor has sizes(x,y) and
    // strides (1,y) It won't work because we except strides (y,1).
    // The trick here is to reverse dimensions, but not the data!
    // This way after transposition actuall data layout will match synapse
    // requirements
    auto maybe_tensor =
        synapse_helpers::tensor_builder(
            {mat2.sizes().rbegin(), mat2.sizes().rend()},
            habana_helpers::pytorch_to_synapse_type(mat2.scalar_type()))
            .mark_persistence(true)
            .build(
                synapse_helpers::HPURegistrar::get_device(device_id),
                graph_handle);
    return absl::get<synapse_helpers::tensor>(std::move(maybe_tensor));
  }
}

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

    syn_helper_inputs.push_back(
        create_hacked_matmul_tensor(mat1, graph_handle, device_id));
    syn_inputs.push_back(syn_helper_inputs[syn_helper_inputs.size() - 1].get());
    syn_helper_inputs.push_back(
        create_hacked_matmul_tensor(mat2, graph_handle, device_id));
    syn_inputs.push_back(syn_helper_inputs[syn_helper_inputs.size() - 1].get());

    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output}, graph_handle, true);

    const std::string node_type = "gemm";
    { // add node
      synGEMMParams params{!mat1.is_contiguous(), !mat2.is_contiguous()};

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

    syn_helper_inputs.push_back(
        create_hacked_matmul_tensor(mat1, graph_handle, device_id));
    syn_inputs.push_back(syn_helper_inputs[syn_helper_inputs.size() - 1].get());
    syn_helper_inputs.push_back(
        create_hacked_matmul_tensor(mat2, graph_handle, device_id));
    syn_inputs.push_back(syn_helper_inputs[syn_helper_inputs.size() - 1].get());
    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(bias, graph_handle, true));
    syn_inputs.push_back(syn_helper_inputs[syn_helper_inputs.size() - 1].get());

    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output}, graph_handle, true);
    std::tie(syn_tmp_helper_tensors, syn_tmp_tensors) =
        habana_helpers::create_tensors(
            std::vector<const at::Tensor*>{&output}, graph_handle, false);

    const std::string node_type1 = "gemm";
    const std::string node_type2 =
        "add_fwd_" + habana_helpers::name_suffix_from_type(mat1.scalar_type());
    { // add node
      synGEMMParams params{!mat1.is_contiguous(), !mat2.is_contiguous()};

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
  check_matmul_params(mat1, mat2, c10::nullopt);
  auto output = at::empty({mat1.size(0), mat2.size(1)}, mat1.options());
  synapse_matmul(output, mat1, mat2);

  // TODO: remove . Workaround for SW-9962
  if (!mat1.is_contiguous())
    adjust_output_tensor_(output);

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
  check_matmul_params(mat1, mat2, &self);
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

  // Note: bias expanded has rank equal to output, but we don't need to actually
  // broadcast data. Putting ones in additional dimensions is enaugh for synapse
  // to handle bcast for us.
  // I am not sure if changing sizes here (tensor metadata) is safe. If
  // something will fail because of that, than you should try to just
  // copy tensor (tensor is metada, not storage) and changed dimensions
  // of copy. Another option is just handling this case inside
  // synapse_matmul
  Tensor bias_expanded;
  auto bias_expanded_sizes = std::vector<int64_t>(output.ndimension(), 1);
  bias_expanded_sizes[output.ndimension() - 1] = self.sizes()[0];
  std::tie(bias_expanded) =
      at::expand_size(self, bias_expanded_sizes, "matmul_with_bias_hpu");
  synapse_matmul(output, mat1, mat2, bias_expanded, beta, alpha);

  // TODO: remove . Workaround for SW-9962
  if (!mat1.is_contiguous())
    adjust_output_tensor_(output);

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
