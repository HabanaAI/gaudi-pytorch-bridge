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
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;

static void check_matmul_params(
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

/*****************************************************************************************************
 * @brief asserts the validity of batched gemm parameters
 * @param[in] mat1 - first matrix
 * @param[in] mat2 - second matrix
 * @param[in] bias - optional bias parameter for affine transformation
 *****************************************************************************************************/
static void check_bmm_matmul_params(
    const Tensor& mat1,
    const Tensor& mat2,
    c10::optional<const at::Tensor*> bias) {
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
}

// output = mat1 x mat2
void synapse_matmul(
    const Tensor& output,
    const Tensor& mat1,
    const Tensor& mat2) {
  PT_KERNEL_BEGIN;
  const auto device_id = mat1.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::string node_type = "gemm";
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(output), IValue(mat1), IValue(mat2)};
  size_t key = habana_helpers::getRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{mat1, mat2};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    habana_helpers::execute_recipe(
        {mat1.data_ptr(), mat2.data_ptr()},
        {output.data_ptr()},
        pt_inputs,
        device_id,
        key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // graph_handle scope
    auto graph = habana_helpers::create_graph(device_id, node_type);
    { // tensors scope
      std::vector<synapse_helpers::tensor> syn_helper_inputs,
          syn_helper_outputs;
      std::vector<synTensor> syn_inputs, syn_outputs;

      syn_helper_inputs.push_back(
          habana_helpers::create_tensor(mat1, graph.get_graph_handle(), true));
      syn_inputs.push_back(
          syn_helper_inputs[syn_helper_inputs.size() - 1].get());
      syn_helper_inputs.push_back(
          habana_helpers::create_tensor(mat2, graph.get_graph_handle(), true));
      syn_inputs.push_back(
          syn_helper_inputs[syn_helper_inputs.size() - 1].get());

      std::tie(syn_helper_outputs, syn_outputs) =
          habana_helpers::create_tensors(
              std::vector<at::Tensor>{output}, graph.get_graph_handle(), true);

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
          pt_inputs,
          device_id,
          key);
    }
  }
}

void habana::MMOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  auto mat1 = inputs[0].toTensor();
  auto mat2 = inputs[1].toTensor();
  check_matmul_params(mat1, mat2, c10::nullopt);
  auto output = habana_helpers::createPTTensor(
      mat1,
      {mat1.size(0), mat2.size(1)},
      mat1.options(),
      mat1.suggest_memory_format(),
      is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  synGEMMParams params{false, false};
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/*****************************************************************************************************
 *@brief Implements torch.mm(mat1, mat2) → Tensor
 *@param mat1 : the first matrix to be multiplied
 *@param mat2 : the second matrix to be multiplied
 *****************************************************************************************************/
at::Tensor mm_hpu(const at::Tensor& mat1, const at::Tensor& mat2) {
  PT_KERNEL_BEGIN;

  const auto device_id = mat1.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::string node_type = "gemm";
  torch::jit::Stack stack = {c10::IValue(mat1), c10::IValue(mat2)};
  habana::MMOperator op(device_id);
  size_t key = op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> inputs = {mat1, mat2};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty({mat1.size(0), mat2.size(1)}, mat1.options());
    op.SetPTInputs(inputs);
    op.SetPTOutput(output);
    op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);

    op.AllocateSynapseInputs(graph, inputs, true);

    op.AllocateAndAddSynapseNode(graph, stack, true);

    op.Compile(graph);
  }
  std::vector<at::Tensor> out = op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

void habana::AddmmOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for addmm operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg0 expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg1 expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg2 expected to be tensor");
  TORCH_CHECK(inputs[3].isScalar(), "Input arg3 expected to be scalar");
  TORCH_CHECK(inputs[4].isScalar(), "Input arg4 expected to be scalar");

  auto self = inputs[0].toTensor();
  auto mat1 = inputs[1].toTensor();
  auto mat2 = inputs[2].toTensor();
  auto beta = inputs[3].toScalar();
  auto alpha = inputs[4].toScalar();

  // TODO: implement support for non-default scalars
  TORCH_CHECK(
      beta.to<int>() == 1,
      "matmul_with_bias_hpu doesn't support non-default scalars yet");
  TORCH_CHECK(
      alpha.to<int>() == 1,
      "matmul_with_bias_hpu doesn't support non-default scalars yet");

  auto device_id = p_context_->device_id_;

  habana::MMOperator mm_op(device_id);
  {
    // input1 = mat1, input2 = mat2
    auto& syn_arg1 =
        mm_op.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
    auto& syn_arg2 =
        mm_op.SetSynapseInput(std::move(p_context_->syn_inputs_[2]));
    torch::jit::Stack stack1 = {c10::IValue(mat1), c10::IValue(mat2)};
    mm_op.AllocateAndAddSynapseNode(graph, stack1, false);
    // Restore original syn_inputs because these will be used in compile in
    // eager mode
    p_context_->syn_inputs_[1] = std::move(syn_arg1);
    p_context_->syn_inputs_[2] = std::move(syn_arg2);
  }

  habana::AddOperator add_op(device_id, mat1.scalar_type());
  {
    // input1 = mat1, input2 = mat2
    auto& syn_arg1 =
        add_op.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    UNUSED auto& syn_arg2 =
        add_op.SetSynapseInput(std::move(mm_op.GetSynOutputs()[0]));
    torch::jit::Stack stack1 = {c10::IValue(self),
                                c10::IValue(mm_op.GetOutputs()[0]),
                                c10::IValue(c10::Scalar(1.0))};
    add_op.AllocateAndAddSynapseNode(graph, stack1, is_output_persistent);
    // Restore original syn_inputs because these will be used in compile in
    // eager mode
    p_context_->syn_inputs_[0] = std::move(syn_arg1);
  }

  p_context_->syn_outputs_.emplace_back(std::move(add_op.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(add_op.GetOutputs()[0]));
}

/*****************************************************************************************************
 *@brief Implements torch.addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None)
 *→ Tensor
 *@param self : matrix to be added
 *@param mat1 : the first matrix to be multiplied
 *@param mat2 : the second matrix to be multiplied
 *@param beta : multiplier for input (β)
 *@param alpha : multiplier for mat1 @ mat2mat1@mat2 (α)
 *****************************************************************************************************/
Tensor addmm_hpu(
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

  const auto device_id = mat1.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  ScalarType scalar_type = mat1.scalar_type();

  std::string node_type =
      "gemm_add_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  habana::AddmmOperator op(device_id, scalar_type);

  std::vector<at::Tensor> inputs = {self, mat1, mat2};
  torch::jit::Stack stack = {
      IValue(self), IValue(mat1), IValue(mat2), IValue(beta), IValue(alpha)};

  size_t key = op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty({mat1.size(0), mat2.size(1)}, mat1.options());
    op.SetPTInputs(inputs);
    op.SetPTOutput(output);
    op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    op.AllocateSynapseInputs(graph, inputs, true);
    op.AllocateAndAddSynapseNode(graph, stack, true);
    op.Compile(graph);
  }
  std::vector<at::Tensor> out = op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

void habana::BmmOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for BmmOut operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");

  auto out = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto mat2 = inputs[2].toTensor();

  check_bmm_matmul_params(self, mat2, c10::nullopt);

  AllocateSynapseOutput(graph, out, is_output_persistent);
  synGEMMParams params{false, false};
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/*****************************************************************************************************
 * @brief Implements batched matrix multiplication _out version
 * @param[in] self - First Tensor, 3D, NHW, bf16/FP32
 * @param[in] mat2 - Second Tensor, 3D, NWC, bf16/FP32
 * @param[in,out] out - Result tensor, 3D, NHC, bf16/FP32
 *****************************************************************************************************/
Tensor& batch_gemm_out_hpu(
    Tensor& out,
    const Tensor& self,
    const Tensor& mat2) {
  PT_KERNEL_BEGIN;

  const auto device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::string node_type = "batch_gemm";
  torch::jit::Stack stack = {IValue(out), IValue(self), IValue(mat2)};
  habana::BmmOutOperator op(device_id, self.scalar_type());
  std::vector<at::Tensor> pt_inputs{self, mat2};

  size_t key = op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    op.SetPTInputs(pt_inputs);
    op.SetPTOutput(out);
    op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    op.AllocateSynapseInputs(graph, pt_inputs, true);
    op.AllocateAndAddSynapseNode(graph, stack, true);
    op.Compile(graph);
  }
  std::vector<at::Tensor> output = op.GetOutputs();
  TORCH_CHECK(output.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return output.at(0);
}

void habana::BmmOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2, "Incorrect size of inputs expected for Bmm operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");

  auto self = inputs[0].toTensor();
  auto mat2 = inputs[1].toTensor();

  auto self_sizes = self.sizes();
  auto mat2_sizes = mat2.sizes();
  auto output = habana_helpers::createPTTensor(
      self,
      {self_sizes[0], self_sizes[1], mat2_sizes[2]},
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
  inputs.insert(inputs.begin(), IValue(output));

  habana::BmmOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

/*****************************************************************************************************
 * @brief Implements batched matrix multiplication
 * @param[in] self - First Tensor, 3D, NHW, bf16/FP32
 * @param[in] mat2 - Second Tensor, 3D, NWC, bf16/FP32
 * @param[out] output - Result tensor, 3D, NHC, bf16/FP32
 *****************************************************************************************************/

Tensor batch_gemm_hpu(const Tensor& self, const Tensor& mat2) {
  PT_KERNEL_BEGIN;

  const auto device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::string node_type = "batch_gemm";
  torch::jit::Stack stack = {IValue(self), IValue(mat2)};
  habana::BmmOperator op(device_id, self.scalar_type());
  std::vector<at::Tensor> pt_inputs{self, mat2};

  size_t key = op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto self_sizes = self.sizes();
    auto mat2_sizes = mat2.sizes();
    auto out = at::empty(
        {self_sizes[0], self_sizes[1], mat2_sizes[2]}, self.options());
    op.SetPTInputs(pt_inputs);
    op.SetPTOutput(out);
    op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    op.AllocateSynapseInputs(graph, pt_inputs, true);
    op.AllocateAndAddSynapseNode(graph, stack, true);
    op.Compile(graph);
  }
  std::vector<at::Tensor> output = op.GetOutputs();
  TORCH_CHECK(output.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return output.at(0);
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

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::mm",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::MMOperator>(device_id);
            })
        .add(
            "aten::addmm",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::AddmmOperator>(
                  device_id, node_type);
            })
        .add("aten::bmm", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<habana::BmmOperator>(device_id, node_type);
        });

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema("aten::mm(Tensor self, Tensor mat2) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(mm_hpu), &mm_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta = 1, Scalar alpha = 1) ->Tensor")
                .impl_unboxedOnlyKernel<decltype(addmm_hpu), &addmm_hpu>(
                    DispatchKey::HABANATensorId)
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
