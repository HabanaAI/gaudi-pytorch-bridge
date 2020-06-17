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
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_device/tensor_builder.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "kernel_utils.h"
#include "linear_kernels.h"

using namespace torch;

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

// output = mat1 x mat2
void synapse_matmul(
    const Tensor& output,
    const Tensor& mat1,
    const Tensor& mat2) {
  const auto device_id = mat1.device().index();
  std::string node_type = "gemm";
  // graph_handle scope
  auto graph = habana_helpers::create_graph(device_id, node_type);

  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs;
    std::vector<synTensor> syn_inputs, syn_outputs;

    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(mat1, graph.get_graph_handle(), true));
    syn_inputs.push_back(syn_helper_inputs[syn_helper_inputs.size() - 1].get());
    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(mat2, graph.get_graph_handle(), true));
    syn_inputs.push_back(syn_helper_inputs[syn_helper_inputs.size() - 1].get());

    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output},
        graph.get_graph_handle(),
        true);

    { // add node
      synGEMMParams params{0, 0};

      graph.add_node(
          std::move(syn_inputs),
          std::move(syn_outputs),
          (void*)&params,
          sizeof(params),
          std::move(node_type));
    }

    habana_helpers::compile_and_run(
        std::move(graph),
        habana_helpers::names(syn_helper_inputs),
        habana_helpers::names(syn_helper_outputs),
        {mat1.data_ptr(), mat2.data_ptr()},
        {output.data_ptr()},
        device_id);
  }
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
  std::string node_type1 = "gemm";
  std::string node_type2 =
      "add_fwd_" + habana_helpers::name_suffix_from_type(mat1.scalar_type());
  // graph_handle scope
  auto graph = habana_helpers::create_graph(device_id, node_type1 + node_type2);
  { // tensors scope
    std::vector<synapse_helpers::tensor> syn_helper_inputs, syn_helper_outputs,
        syn_tmp_helper_tensors;
    std::vector<synTensor> syn_inputs, syn_outputs, syn_tmp_tensors;

    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(mat1, graph.get_graph_handle(), true));
    syn_inputs.push_back(syn_helper_inputs[syn_helper_inputs.size() - 1].get());
    syn_helper_inputs.push_back(
        habana_helpers::create_tensor(mat2, graph.get_graph_handle(), true));
    syn_inputs.push_back(syn_helper_inputs[syn_helper_inputs.size() - 1].get());

    std::tie(syn_helper_outputs, syn_outputs) = habana_helpers::create_tensors(
        std::vector<const at::Tensor*>{&output},
        graph.get_graph_handle(),
        true);
    std::tie(syn_tmp_helper_tensors, syn_tmp_tensors) =
        habana_helpers::create_tensors(
            std::vector<const at::Tensor*>{&output},
            graph.get_graph_handle(),
            false);

    { // add node
      synGEMMParams params{0, 0};

      graph.add_node(
          std::move(syn_inputs),
          std::move(syn_tmp_tensors),
          (void*)&params,
          sizeof(params),
          std::move(node_type1));
    }
    { // add node
      syn_helper_inputs.push_back(
          habana_helpers::create_tensor(bias, graph.get_graph_handle(), true));
      syn_inputs.push_back(
          syn_helper_inputs[syn_helper_inputs.size() - 1].get());
      syn_tmp_tensors.push_back(syn_inputs[2]); // mm_out + bias
      graph.add_node(
          std::move(syn_tmp_tensors),
          std::move(syn_outputs),
          nullptr,
          0,
          std::move(node_type2));
    }
    habana_helpers::compile_and_run(
        std::move(graph),
        habana_helpers::names(syn_helper_inputs),
        habana_helpers::names(syn_helper_outputs),
        {mat1.data_ptr(), mat2.data_ptr(), bias.data_ptr()},
        {output.data_ptr()},
        device_id);
  }
}

habana::MMOperator::MMOperator(int device_id): HabanaOperator("gemm") {
  this->CreateSynContext(device_id);
  kernel_meta_data_.input_layout.assign({LayoutFormat::ANY, LayoutFormat::ANY});
  kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
}

void habana::MMOperator::AllocateAndAddSynapseNode(synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK( inputs.size() == 2,
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  auto mat1 = inputs[0].toTensor();
  auto mat2 = inputs[1].toTensor();
  check_matmul_params(mat1, mat2, c10::nullopt);
  auto output = at::empty({mat1.size(0), mat2.size(1)}, mat1.options());
  AllocateSynapseOutput(graph, output, is_output_persistent);
  synGEMMParams params{false, false};
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

at::Tensor matmul_hpu(const at::Tensor& mat1, const at::Tensor& mat2) {
  PT_KERNEL_BEGIN;

  const auto device_id = mat1.device().index();
  std::string node_type = "gemm";

  auto graph = habana_helpers::create_graph(device_id, node_type);

  habana::MMOperator op(device_id);

  std::vector<const at::Tensor*> inputs = {&mat1, &mat2};
  torch::jit::Stack stack = {c10::IValue(mat1), c10::IValue(mat2)};

  op.AllocateSynapseInputs(graph, inputs, true);

  op.AllocateAndAddSynapseNode(graph, stack, true);

  op.Compile(graph);
  std::vector<at::Tensor> out = op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

Tensor matmul_with_bias_hpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  PT_KERNEL_BEGIN;
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

  PT_KERNEL_END;
  return output;
}

/*****************************************************************************************************
 * @brief asserts the validity of batched gemm parameters
 * @param[in] mat1 - first matrix
 * @param[in] mat2 - second matrix
 * @param[in] bias - optional bias parameter for affine transformation
 *****************************************************************************************************/
void check_bmm_matmul_params(
    const Tensor& mat1,
    const Tensor& mat2,
    c10::optional<const at::Tensor*> bias) {
  PT_KERNEL_BEGIN;
  TORCH_CHECK(mat1.ndimension() == 3, "Batched gemm supports only 3d matrices");
  TORCH_CHECK(mat2.ndimension() == 3, "Batched gemm supports only 3d matrices");
  TORCH_CHECK(
      mat1.size(2) == mat2.size(1), "matmul inner dimensions doesn't match");
  TORCH_CHECK(
      static_cast<int>(mat1.is_contiguous()) + mat2.is_contiguous() == 2,
      "Both matrices should be contiguous.",
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
  PT_KERNEL_END;
}

/*****************************************************************************************************
 * @brief Implements batched matrix multiplication _out version
 * @param[in] self - First Tensor, 3D, NHW, bf16/FP32
 * @param[in] mat2 - Second Tensor, 3D, NWC, bf16/FP32
 * @param[in,out] out - Result tensor, 3D, NHC, bf16/FP32
 *****************************************************************************************************/
void batch_gemm_out_hpu(Tensor& out, const Tensor& self, const Tensor& mat2) {
  PT_KERNEL_BEGIN;

  check_bmm_matmul_params(self, mat2, c10::nullopt);

  std::vector<const at::Tensor*> pt_inputs{&self, &mat2};
  std::vector<const at::Tensor*> pt_outputs{&out};

  synGEMMParams params{0, 0};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "batch_gemm",
      &params,
      sizeof(params),
      SynapsePassType::NO_PASS);

  PT_KERNEL_END;
}

/*****************************************************************************************************
 * @brief Implements batched matrix multiplication
 * @param[in] self - First Tensor, 3D, NHW, bf16/FP32
 * @param[in] mat2 - Second Tensor, 3D, NWC, bf16/FP32
 * @param[out] output - Result tensor, 3D, NHC, bf16/FP32
 *****************************************************************************************************/

Tensor batch_gemm_hpu(const Tensor& self, const Tensor& mat2) {
  PT_KERNEL_BEGIN;

  // If input is a b×n×m tensor, mat2 is a b×m×p tensor, out will be a b×n×p
  // tensor
  auto self_sizes = self.sizes();
  auto mat2_sizes = mat2.sizes();
  auto out =
      at::empty({self_sizes[0], self_sizes[1], mat2_sizes[2]}, self.options());
  batch_gemm_out_hpu(out, self, mat2);

  PT_KERNEL_END;

  return out;
}

/*****************************************************************************************************
*@brief Implements torch.dot(vector,vector)
self - 1D m
other - 1D m
output - 0-D tensor
*****************************************************************************************************/
Tensor dot_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  Tensor output = at::empty({1, 1}, self.options());

  // synapse expects 2-D matrices
  auto self_hpu = self.view({1, self.sizes()[0]});
  auto other_hpu = other.view({other.sizes()[0], 1});

  synapse_matmul(output, self_hpu, other_hpu);

  // PT expects 0-D
  output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});

  PT_KERNEL_END;
  return output;
}

/*****************************************************************************************************
*@brief Implements torch.mv(tensor,vector)
self - 2D nxm
other - 1D m
output - 1D n
*****************************************************************************************************/
Tensor mv_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  auto self_sizes = self.sizes();
  auto other_sizes = other.sizes();

  // synapse expects 2-D matrices
  Tensor output = at::empty({self_sizes[0], 1}, self.options());
  auto other_hpu = other.view({other_sizes[0], 1});

  synapse_matmul(output, self, other_hpu);

  // PT expects 1-D
  output = output.view(-1);

  PT_KERNEL_END;
  return output;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema("aten::mm(Tensor self, Tensor mat2) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(matmul_hpu), &matmul_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta = 1, Scalar alpha = 1) ->Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(matmul_with_bias_hpu),
                    &matmul_with_bias_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(batch_gemm_out_hpu),
                    &batch_gemm_out_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::bmm(Tensor self, Tensor mat2) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(batch_gemm_hpu),
                    &batch_gemm_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::dot(Tensor self, Tensor tensor) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(dot_hpu), &dot_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::mv(Tensor self, Tensor vec)->Tensor")
                .impl_unboxedOnlyKernel<decltype(mv_hpu), &mv_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
