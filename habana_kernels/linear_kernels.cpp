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
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_kernels/reduction_kernels.h"
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

std::vector<int64_t> habana::MMOperator::compute_output_shape(
    at::Tensor self,
    at::Tensor other) {
  return {self.size(0), other.size(1)};
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
  auto shape_out = habana::MMOperator::compute_output_shape(mat1, mat2);
  auto output = habana_helpers::createPTTensor(
      mat1,
      shape_out,
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
    auto shape_out = habana::MMOperator::compute_output_shape(mat1, mat2);
    auto output = at::empty(shape_out, mat1.options());
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

std::vector<int64_t> habana::BmmOperator::compute_output_shape(
    const Tensor& self,
    const Tensor& mat2) {
  auto self_sizes = self.sizes();
  auto mat2_sizes = mat2.sizes();
  std::vector<int64_t> shape_out = {
      self_sizes[0], self_sizes[1], mat2_sizes[2]};
  return shape_out;
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

  auto shape_out = habana::BmmOperator::compute_output_shape(self, mat2);

  auto output = habana_helpers::createPTTensor(
      self,
      shape_out,
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
    auto shape_out = habana::BmmOperator::compute_output_shape(self, mat2);
    auto out = at::empty(shape_out, self.options());
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
void habana::DotOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  auto mat1 = inputs[0].toTensor(); // size of mat1 is m
  auto mat2 = inputs[1].toTensor(); // size of mat2 is m

  std::vector<c10::IValue> stack;
  // ReShape Operator to covert 1d tensor to 2d for mat1
  int64_t data_m1[2];
  data_m1[0] = 1;
  data_m1[1] = mat1.numel();
  c10::IntArrayRef shape_m1(data_m1, 2);
  ReshapeOperator ReShapeOp_m1(
      this->p_context_->device_id_, mat1.scalar_type());
  auto& reShape_syn_m1 =
      ReShapeOp_m1.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  // Build Params for the graph
  stack.emplace_back(IValue(mat1));
  stack.emplace_back(IValue(shape_m1));
  ReShapeOp_m1.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[0] = std::move(reShape_syn_m1);
  stack.clear();

  // ReShape Operator to covert 1d tensor to 2d for mat2
  int64_t data_m2[2];
  data_m2[0] = mat2.numel();
  data_m2[1] = 1;
  c10::IntArrayRef shape_m2(data_m2, 2);
  // Create the operator
  ReshapeOperator ReShapeOp_m2(
      this->p_context_->device_id_, mat2.scalar_type());
  auto& reShape_syn_m2 =
      ReShapeOp_m2.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  // Build Params for the graph
  stack.emplace_back(IValue(mat2));
  stack.emplace_back(IValue(shape_m2));
  ReShapeOp_m2.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[1] = std::move(reShape_syn_m2);
  stack.clear();

  // Matmul Operator (1xn) * (nx1) = (1x1)
  MMOperator mmOp(this->p_context_->device_id_);
  mmOp.SetSynapseInput(std::move(ReShapeOp_m1.GetSynOutputs()[0]));
  mmOp.SetSynapseInput(std::move(ReShapeOp_m2.GetSynOutputs()[0]));
  // Build Params for the graph
  stack.emplace_back(IValue(ReShapeOp_m1.GetOutputs()[0]));
  stack.emplace_back(IValue(ReShapeOp_m2.GetOutputs()[0]));
  mmOp.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // ReShape Operator to covert 2d tensor to 1d for output
  int64_t data[1];
  data[0] = mmOp.GetOutputs()[0].numel();
  c10::IntArrayRef shape(data, 1);
  ReshapeOperator ReShapeOp_out(
      this->p_context_->device_id_, mmOp.GetOutputs()[0].scalar_type());
  ReShapeOp_out.SetSynapseInput(std::move(mmOp.GetSynOutputs()[0]));
  // Build Params for the graph
  stack.emplace_back(IValue(mmOp.GetOutputs()[0]));
  stack.emplace_back(IValue(shape));
  ReShapeOp_out.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

  p_context_->syn_outputs_.emplace_back(
      std::move(ReShapeOp_out.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(
      std::move(ReShapeOp_out.GetOutputs()[0]));
}

Tensor dot_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  const auto device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::string node_type = "dot";
  torch::jit::Stack stack = {c10::IValue(self), c10::IValue(other)};
  habana::DotOperator op(device_id);
  size_t key = op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> inputs = {self, other};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty({1, 1}, self.options());
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

  // PT expects 0-D
  out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});

  PT_KERNEL_END;
  return out.at(0);
}

/*****************************************************************************************************
*@brief Implements torch.mv(tensor,vector)
self - 2D nxm
other - 1D m
output - 1D n
*****************************************************************************************************/
void habana::MvOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  auto mat1 = inputs[0].toTensor(); // mxn
  auto mat2 = inputs[1].toTensor(); // size of mat2 is n

  // ReShape Operator to covert n to nx1 for mat2
  int64_t data[2];
  data[0] = mat2.numel();
  data[1] = 1;
  c10::IntArrayRef shape(data, 2);
  // Create the operator
  ReshapeOperator ReShapeOp(this->p_context_->device_id_, mat2.scalar_type());
  auto& reShape_syn =
      ReShapeOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  // Build Params for the graph
  std::vector<c10::IValue> stack{IValue(mat2), IValue(shape)};
  ReShapeOp.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[1] = std::move(reShape_syn);
  stack.clear();

  // Matmul Operator (mxn) * (nx1) = (mx1)
  MMOperator mmOp(this->p_context_->device_id_);
  auto& mm_syn_1 = mmOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  mmOp.SetSynapseInput(std::move(ReShapeOp.GetSynOutputs()[0]));
  // Build Params for the graph
  stack.emplace_back(IValue(mat1));
  stack.emplace_back(IValue(ReShapeOp.GetOutputs()[0]));
  mmOp.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[0] = std::move(mm_syn_1);
  stack.clear();

  // PT expects 1-D
  // ReShape Operator to covert mx1 to 1xm for output of Matmul Operator
  int64_t data2[1];
  data2[0] = mmOp.GetOutputs()[0].numel();
  c10::IntArrayRef shape2(data2, 1);
  ReshapeOperator ReShapeOp_2(
      this->p_context_->device_id_, mmOp.GetOutputs()[0].scalar_type());
  ReShapeOp_2.SetSynapseInput(std::move(mmOp.GetSynOutputs()[0]));
  // Build Params for the graph
  stack.emplace_back(IValue(mmOp.GetOutputs()[0]));
  stack.emplace_back(IValue(shape2));
  ReShapeOp_2.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

  p_context_->syn_outputs_.emplace_back(
      std::move(ReShapeOp_2.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(ReShapeOp_2.GetOutputs()[0]));
}

Tensor mv_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  const auto device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::string node_type = "mv";
  torch::jit::Stack stack = {c10::IValue(self), c10::IValue(other)};
  habana::MvOperator op(device_id);
  size_t key = op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> inputs = {self, other};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty({1, self.size(0)}, self.options());
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

std::vector<int64_t> habana::MatMulOperator::compute_output_shape(
    const at::Tensor &tensor1,
    const at::Tensor &tensor2) {
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {

    auto size1 = tensor1.sizes();
    auto size2 = dim_tensor2 == 1 ?
        at::infer_size({-1, 1}, tensor2.numel()) : tensor2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    if (dim_tensor2 > 1) {
      output_size.push_back(size2[dim_tensor2 - 1]);
    }
    return output_size;
  }
  else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) &&
      (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;

    IntArrayRef batch_tensor1(
        tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    int64_t p = tensor2.size(-1);
    IntArrayRef batch_tensor2(
        tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion =
        at::infer_size(batch_tensor1, batch_tensor2);

    // reshape batches back into result
    std::vector<int64_t> output_size(expand_batch_portion);
    if (dim_tensor1 > 1) {
      output_size.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_size.push_back(p);
    }
    return output_size;
  }
  else {
    TORCH_CHECK(false, "don't support matmul of this size");
    return std::vector<int64_t>();
  }
}

void habana::MatMulOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for matmul operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  auto tensor1 = inputs[0].toTensor();
  auto tensor2 = inputs[1].toTensor();

  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  //matmul is supoorted only for BERT Large with following dims
  //(dim1 >=3 && (dim1==2 || dim2==2)
  //((dim1 >=1 && dim2 >=1) && (dim1>=3 || dim2>=3))

  auto output_shape = MatMulOperator::compute_output_shape(tensor1, tensor2);

  if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    // optimization: use mm instead of bmm by folding tensor1's batch into
    // its leading matrix dimension.
    auto size1 = tensor1.sizes();
    auto inferred_ten1_dims =
        at::infer_size({-1, size1[size1.size() - 1]}, tensor1.numel());

    ReshapeOperator reshape_ten1(tensor1.device().index(), tensor1.scalar_type());

    auto& syn_input1 =
        reshape_ten1.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    torch::jit::Stack stack =
        {c10::IValue(tensor1), c10::IValue(inferred_ten1_dims)};
    reshape_ten1.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[0] = std::move(syn_input1);
    Tensor t1 = reshape_ten1.GetOutputs()[0];
    stack.clear();

    Tensor t2 = tensor2;
    bool is_reshaped = false;
    // Add Reshape node to graph
    ReshapeOperator reshape_ten2(tensor2.device().index(), tensor2.scalar_type());
    if (dim_tensor2 == 1) {
      auto inferred_ten2_dims = at::infer_size({-1, 1}, tensor2.numel());
      auto& syn_input2 =
          reshape_ten2.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
      torch::jit::Stack stack =
          {c10::IValue(tensor2), c10::IValue(inferred_ten2_dims)};
      reshape_ten2.AllocateAndAddSynapseNode(graph, stack, false);
      p_context_->syn_inputs_[1] = std::move(syn_input2);
      t2 = reshape_ten2.GetOutputs()[0];
      is_reshaped = true;
    }

    MMOperator mm_op(tensor1.device().index());
    // input1 = mat1, input2 = mat2
    mm_op.SetSynapseInput(std::move(reshape_ten1.GetSynOutputs()[0]));
    auto &syn_input2 = mm_op.SetSynapseInput(
        std::move( is_reshaped ? reshape_ten2.GetSynOutputs()[0] :
            p_context_->syn_inputs_[1]));
    stack = {c10::IValue(t1), c10::IValue(t2)};
    mm_op.AllocateAndAddSynapseNode(graph, stack, false);
    if(!is_reshaped) {
      p_context_->syn_inputs_[1] = std::move(syn_input2);
    }
    stack.clear();
    //reshape the output
    auto output = mm_op.GetOutputs()[0];
    ReshapeOperator reshape_out(tensor2.device().index(), tensor2.scalar_type());
    reshape_out.SetSynapseInput(std::move(mm_op.GetSynOutputs()[0]));
    stack = {c10::IValue(output), c10::IValue(output_shape)};
    reshape_out.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    p_context_->syn_outputs_.emplace_back(
        std::move(reshape_out.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(reshape_out.GetOutputs()[0]));
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) &&
        (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
      // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
      // we track m1 vs m2 separately even though they must match for nicer error messages
      int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
      int64_t m1 = tensor1.size(-1);
      IntArrayRef batch_tensor1(
          tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
      int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
      int64_t p = tensor2.size(-1);
      IntArrayRef batch_tensor2(
          tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

      // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
      std::vector<int64_t> expand_batch_portion =
          at::infer_size(batch_tensor1, batch_tensor2);

      std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
      tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

      std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
      tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

      int expand_batch_product = std::accumulate(
          expand_batch_portion.begin(),
          expand_batch_portion.end(),
          1,
          std::multiplies<int64_t>());

      std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
      tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

      std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
      tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});

      BroadcastOperator bcastOpTens1(
          tensor1.device().index(), tensor1.scalar_type());
      torch::jit::Stack stack =
          {IValue(tensor1),
          IValue(tensor1_expand_size),
          IValue(false)};
      auto& syn_input1 =
          bcastOpTens1.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
      bcastOpTens1.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();
      p_context_->syn_inputs_[0] = std::move(syn_input1);
      //expand tensor2
      BroadcastOperator bcastOpTens2(
        tensor2.device().index(), tensor2.scalar_type());
      stack =
          {IValue(tensor2), IValue(tensor2_expand_size), IValue(false)};
      auto& syn_input2 =
          bcastOpTens2.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
      bcastOpTens2.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();
      p_context_->syn_inputs_[1] = std::move(syn_input2);

      ReshapeOperator reshape_ten1(tensor1.device().index(), tensor1.scalar_type());
      reshape_ten1.SetSynapseInput(std::move(bcastOpTens1.GetSynOutputs()[0]));
      stack =
          {IValue(bcastOpTens1.GetOutputs()[0]),
          IValue(tensor1_bmm_view)};
      reshape_ten1.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      ReshapeOperator reshape_ten2(tensor2.device().index(), tensor2.scalar_type());
      reshape_ten2.SetSynapseInput(std::move(bcastOpTens2.GetSynOutputs()[0]));
      stack =
          {IValue(bcastOpTens2.GetOutputs()[0]),
          IValue(tensor2_bmm_view)};
      reshape_ten2.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      BmmOperator bmm_op(tensor1.device().index(), tensor1.scalar_type());

      bmm_op.SetSynapseInput(std::move(reshape_ten1.GetSynOutputs()[0]));
      bmm_op.SetSynapseInput(std::move(reshape_ten2.GetSynOutputs()[0]));
      stack =
          {IValue(reshape_ten1.GetOutputs()[0]),
          IValue(reshape_ten2.GetOutputs()[0])};
      bmm_op.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

    //reshape the output
    auto output = bmm_op.GetOutputs()[0];
    ReshapeOperator reshape_out(tensor2.device().index(), tensor2.scalar_type());
    reshape_out.SetSynapseInput(std::move(bmm_op.GetSynOutputs()[0]));
    stack = {IValue(output), IValue(output_shape)};
    reshape_out.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    p_context_->syn_outputs_.emplace_back(
        std::move(reshape_out.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(reshape_out.GetOutputs()[0]));
  }
  else {
     PT_KERNEL_FATAL("matmul with following dimension is not supported ",
          dim_tensor1, "D and ", dim_tensor2, "D");
  }
}

Tensor matmul_hpu(const Tensor& tensor1, const Tensor& tensor2) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = tensor1.scalar_type();
  std::string node_type =
      "matmul" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = tensor1.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  habana::MatMulOperator Op(device_id);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(tensor1), IValue(tensor2)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{tensor1, tensor2};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto shape_out = habana::MMOperator::compute_output_shape(tensor1, tensor2);
    auto output = at::empty(shape_out, tensor1.options());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

template <typename T>
synapse_helpers::tensor_or_ref habana::MatmulBackwardOperator::MatBwTranspose(
    synapse_helpers::graph& graph,
    T& Op,
    Tensor& mat,
    synapse_helpers::tensor_or_ref syn_input) {
  auto dim = mat.dim();
  if (dim == 1) {
    // Add a memcpy node to graph (to get out = mat)
    auto& op_syn_input = Op.SetSynapseInput(std::move(syn_input));
    torch::jit::Stack stack = {IValue(mat)};
    Op.AllocateAndAddSynapseNode(graph, stack, false);
    syn_input = std::move(op_syn_input);
  } else if (dim == 2) {
    // Add a transpose node to graph
    auto& op_syn_input = Op.SetSynapseInput(std::move(syn_input));
    torch::jit::Stack stack = {IValue(mat), IValue(0), IValue(1)};
    Op.AllocateAndAddSynapseNode(graph, stack, false);
    syn_input = std::move(op_syn_input);
  } else {
    // Add a permute node to graph (since permute is only on last
    // 2 dims we use transpose node to simplify code). Keep the
    // commented code below, just in case we need to go back to
    // permute again for some reason.

    // std::vector<int64_t> dims_v(dim);
    // std::iota(std::begin(dims_v), std::end(dims_v), 0);
    // dims_v[dims_v.size() - 1] = dim - 2;
    // dims_v[dims_v.size() - 2] = dim - 1;
    // IntArrayRef dims(dims_v.data(), dims_v.size());

    auto& op_syn_input = Op.SetSynapseInput(std::move(syn_input));
    torch::jit::Stack stack = {IValue(mat), IValue(dim - 2), IValue(dim - 1)};
    Op.AllocateAndAddSynapseNode(graph, stack, false);
    syn_input = std::move(op_syn_input);
  }

  return syn_input;
}

synapse_helpers::tensor_or_ref habana::MatmulBackwardOperator::MatBwReshape(
    synapse_helpers::graph& graph,
    Tensor& mat,
    std::vector<int64_t> sizes,
    synapse_helpers::tensor_or_ref syn_input) {
  ReshapeOperator reshape(mat.device().index(), mat.scalar_type());
  auto& reshape_mat_syn_input = reshape.SetSynapseInput(std::move(syn_input));
  torch::jit::Stack stack = {IValue(mat), IValue(sizes)};
  reshape.AllocateAndAddSynapseNode(graph, stack, false);
  ReshapeOpList.push_back(reshape);
  syn_input = std::move(reshape_mat_syn_input);

  return syn_input;
}

template <typename T>
std::tuple<synapse_helpers::tensor_or_ref, synapse_helpers::tensor_or_ref>
habana::MatmulBackwardOperator::MatBwSpecialFold(
    synapse_helpers::graph& graph,
    T& Op,
    Tensor& mat1,
    Tensor& mat2,
    synapse_helpers::tensor_or_ref syn_input1,
    synapse_helpers::tensor_or_ref syn_input2) {
  /*
    # In matmul backward case of [b, m, n] * [b, n, p] => [m, p],
    # instead of doing [b, m, p] and then reduce to [m, p]
    # whice potentially uses large intermediate of size b*m*p,
    # we do [m, bn] * [bn, p] to avoid having the large
    # intermediate, thus reduces max memory usage.
  */
  TransposeOperator transpose1(mat1.device().index(), mat1.scalar_type());
  syn_input1 = MatBwTranspose(graph, transpose1, mat1, std::move(syn_input1));
  auto transpose_out = transpose1.GetOutputs()[0];

  auto ReshapeListSizeIn = ReshapeOpList.size();

  std::vector<int64_t> reshape_mat1_sizes{
      -1, transpose_out.size(transpose_out.dim() - 1)};
  MatBwReshape(
      graph,
      transpose1.GetOutputs()[0],
      reshape_mat1_sizes,
      std::move(transpose1.GetSynOutputs()[0]));

  std::vector<int64_t> reshape_mat2_sizes{-1, mat2.size(mat2.dim() - 1)};
  syn_input2 =
      MatBwReshape(graph, mat2, reshape_mat2_sizes, std::move(syn_input2));

  TransposeOperator transpose2(mat2.device().index(), mat2.scalar_type());
  UNUSED auto& transpose2_syn_input = transpose2.SetSynapseInput(
      std::move(ReshapeOpList.at(ReshapeListSizeIn).GetSynOutputs()[0]));
  torch::jit::Stack stack = {
      IValue(ReshapeOpList.at(ReshapeListSizeIn).GetOutputs()[0]),
      IValue(0),
      IValue(1)};
  transpose2.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  UNUSED auto& mm_syn_input_0 =
      Op.SetSynapseInput(std::move(transpose2.GetSynOutputs()[0]));
  UNUSED auto& mm_syn_input_1 = Op.SetSynapseInput(
      std::move(ReshapeOpList.at(ReshapeListSizeIn + 1).GetSynOutputs()[0]));
  stack = {IValue(transpose2.GetOutputs()[0]),
           IValue(ReshapeOpList.at(ReshapeListSizeIn + 1).GetOutputs()[0])};
  Op.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  ReshapeOpList.pop_back();
  ReshapeOpList.pop_back();

  return std::make_tuple(std::move(syn_input1), std::move(syn_input2));
}

template <typename T>
std::tuple<synapse_helpers::tensor_or_ref, synapse_helpers::tensor_or_ref>
habana::MatmulBackwardOperator::MatBwSize(
    synapse_helpers::graph& graph,
    T& Op,
    Tensor& mat1,
    Tensor& mat2,
    IntArrayRef sizes,
    synapse_helpers::tensor_or_ref syn_input1,
    synapse_helpers::tensor_or_ref syn_input2,
    bool is_output_persistent) {
  auto dim_out = sizes.size();
  auto dim1 = mat1.dim();
  auto dim2 = mat2.dim();

  if (dim_out == 2 and dim1 == dim2 and dim1 >= 3) {
    /* out = AD_matmul_bw_special_fold(mat1, mat2) */
    habana::MMOperator mm(mat1.device().index());
    std::tie(syn_input1, syn_input2) = MatBwSpecialFold(
        graph, mm, mat1, mat2, std::move(syn_input1), std::move(syn_input2));

    UNUSED auto& gradsum_syn_input_1 =
        Op.SetSynapseInput(std::move(mm.GetSynOutputs()[0]));
    torch::jit::Stack stack = {IValue(mm.GetOutputs()[0]), IValue(sizes)};
    Op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  } else if ((dim1 + dim2) == static_cast<int64_t>(dim_out)) {
    /*
    if dim2 == 1:
        target_dim2 = 0
    else:
        target_dim2 = -2
    out = torch.matmul(mat1.unsqueeze(dim1), mat2.unsqueeze(target_dim2))
    */
    std::vector<int64_t> reshape1_sizes{mat1.sizes().vec()};
    reshape1_sizes.push_back(1);
    syn_input1 =
        MatBwReshape(graph, mat1, reshape1_sizes, std::move(syn_input1));

    std::vector<int64_t> reshape2_sizes{mat2.sizes().vec()};
    reshape2_sizes.insert(reshape2_sizes.cbegin(), 1);
    syn_input2 =
        MatBwReshape(graph, mat2, reshape2_sizes, std::move(syn_input2));

    habana::MatMulOperator matmul(mat1.device().index());
    UNUSED auto& matmul_syn_input_0 = matmul.SetSynapseInput(
        std::move(ReshapeOpList.at(0).GetSynOutputs()[0]));
    UNUSED auto& matmul_syn_input_1 = matmul.SetSynapseInput(
        std::move(ReshapeOpList.at(1).GetSynOutputs()[0]));
    torch::jit::Stack stack = {IValue(ReshapeOpList.at(0).GetOutputs()[0]),
                               IValue(ReshapeOpList.at(1).GetOutputs()[0])};
    matmul.AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    ReshapeOpList.clear();

    UNUSED auto& gradsum_syn_input_0 =
        Op.SetSynapseInput(std::move(matmul.GetSynOutputs()[0]));
    stack = {IValue(matmul.GetOutputs()[0]), IValue(sizes)};
    Op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  } else if ((dim_out == 1) && (dim1 - dim2) == 1 && (dim1 >= 3)) {
    /*
    elif dim_out == 1 and dim1 - dim2 == 1 and dim1 >= 3:
        mat2_unsqueeze = mat2.unsqueeze(-1)
        out = AD_matmul_bw_special_fold(mat1, mat2_unsqueeze)
        out = out.squeeze(-1)
    */
    std::vector<int64_t> reshape2_sizes{mat2.sizes().vec()};
    reshape2_sizes.push_back(1);
    syn_input2 =
        MatBwReshape(graph, mat2, reshape2_sizes, std::move(syn_input2));

    // Using MM directly since MatMul is anyway going to call MM for a 2x2 case
    habana::MMOperator matmul(mat1.device().index());
    std::tie(syn_input1, std::ignore) = MatBwSpecialFold(
        graph,
        matmul,
        mat1,
        ReshapeOpList.at(0).GetOutputs()[0],
        std::move(syn_input1),
        std::move(ReshapeOpList.at(0).GetSynOutputs()[0]));

    std::vector<int64_t> reshape1_sizes{matmul.GetOutputs()[0].sizes().vec()};
    reshape1_sizes.pop_back();
    MatBwReshape(
        graph,
        matmul.GetOutputs()[0],
        reshape1_sizes,
        std::move(matmul.GetSynOutputs()[0]));

    UNUSED auto& gradsum_syn_input_0 =
        Op.SetSynapseInput(std::move(ReshapeOpList.at(1).GetSynOutputs()[0]));
    torch::jit::Stack stack = {IValue(ReshapeOpList.at(1).GetOutputs()[0]), IValue(sizes)};
    Op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    ReshapeOpList.clear();
  } else if (static_cast<int64_t>(dim_out) == (dim1 - dim2)) {
    /* out = torch.matmul(mat1, mat2.unsqueeze(dim2)).squeeze(-1) */
    std::vector<int64_t> reshape2_sizes{mat2.sizes().vec()};
    reshape2_sizes.push_back(1);
    syn_input2 =
        MatBwReshape(graph, mat2, reshape2_sizes, std::move(syn_input2));

    habana::MatMulOperator matmul(mat1.device().index());
    auto& matmul_syn_input_0 = matmul.SetSynapseInput(std::move(syn_input1));
    UNUSED auto& matmul_syn_input_1 = matmul.SetSynapseInput(
        std::move(ReshapeOpList.at(0).GetSynOutputs()[0]));
    torch::jit::Stack stack = {IValue(mat1),
                               IValue(ReshapeOpList.at(0).GetOutputs()[0])};
    matmul.AllocateAndAddSynapseNode(graph, stack, false);
    syn_input1 = std::move(matmul_syn_input_0);
    stack.clear();

    std::vector<int64_t> reshape1_sizes{matmul.GetOutputs()[0].sizes().vec()};
    reshape1_sizes.erase(reshape1_sizes.cbegin());
    MatBwReshape(
        graph,
        matmul.GetOutputs()[0],
        reshape1_sizes,
        std::move(matmul.GetSynOutputs()[0]));

    UNUSED auto& gradsum_syn_input_0 =
        Op.SetSynapseInput(std::move(ReshapeOpList.at(1).GetSynOutputs()[0]));
    stack = {IValue(ReshapeOpList.at(1).GetOutputs()[0]), IValue(sizes)};
    Op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    ReshapeOpList.clear();
  } else if (static_cast<int64_t>(dim_out) == (dim2 - dim1)) {
    /* out = torch.matmul(mat1.unsqueeze(-2), mat2).squeeze(-2) */
    std::vector<int64_t> reshape1_sizes{mat1.sizes().vec()};
    reshape1_sizes.insert(reshape1_sizes.cbegin(), 1);
    syn_input1 =
        MatBwReshape(graph, mat1, reshape1_sizes, std::move(syn_input1));

    habana::MatMulOperator matmul(mat1.device().index());
    UNUSED auto& matmul_syn_input_0 = matmul.SetSynapseInput(
        std::move(ReshapeOpList.at(0).GetSynOutputs()[0]));
    auto& matmul_syn_input_1 = matmul.SetSynapseInput(std::move(syn_input2));
    torch::jit::Stack stack = {IValue(ReshapeOpList.at(0).GetOutputs()[0]),
                               IValue(mat2)};
    matmul.AllocateAndAddSynapseNode(graph, stack, false);
    syn_input2 = std::move(matmul_syn_input_1);
    stack.clear();

    std::vector<int64_t> reshape2_sizes{matmul.GetOutputs()[0].sizes().vec()};
    reshape2_sizes.pop_back();
    MatBwReshape(
        graph,
        matmul.GetOutputs()[0],
        reshape2_sizes,
        std::move(matmul.GetSynOutputs()[0]));

    ReshapeOpList.clear();

    UNUSED auto& gradsum_syn_input_0 =
        Op.SetSynapseInput(std::move(matmul.GetSynOutputs()[0]));
    stack = {IValue(matmul.GetOutputs()[0]), IValue(sizes)};
    Op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  } else {
    /* out = torch.matmul(mat1, mat2) */
    habana::MatMulOperator matmul(mat1.device().index());
    auto& matmul_syn_input_0 = matmul.SetSynapseInput(std::move(syn_input1));
    auto& matmul_syn_input_1 = matmul.SetSynapseInput(std::move(syn_input2));
    torch::jit::Stack stack = {IValue(mat1), IValue(mat2)};
    matmul.AllocateAndAddSynapseNode(graph, stack, false);
    syn_input1 = std::move(matmul_syn_input_0);
    syn_input2 = std::move(matmul_syn_input_1);
    stack.clear();

    UNUSED auto& gradsum_syn_input_0 =
        Op.SetSynapseInput(std::move(matmul.GetSynOutputs()[0]));
    stack = {IValue(matmul.GetOutputs()[0]), IValue(sizes)};
    Op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  }

  return std::make_tuple(std::move(syn_input1), std::move(syn_input2));
}

void habana::MatmulBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for matmul backward operator");

  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input1 type expected to be tensor for matmul backward operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input2 type expected to be tensor for matmul backward operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input3 type expected to be tensor for matmul backward operator");

  auto grad_out = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto other = inputs[2].toTensor();

  // grad_self = AD_matmul_bw_size(grad_output, AD_mat_transpose(other),
  // self_size)._grad_sum_to_size(self_size)
  TransposeOperator transpose(other.device().index(), other.scalar_type());
  MemCopyOperator memcpy(other.device().index(), other.scalar_type());
  if (other.dim() > 1) {
    p_context_->syn_inputs_[2] = MatBwTranspose(
        graph, transpose, other, std::move(p_context_->syn_inputs_[2]));
  } else {
    p_context_->syn_inputs_[2] = MatBwTranspose(
        graph, memcpy, other, std::move(p_context_->syn_inputs_[2]));
  }

  GradSumToSizeOperator gradsum(self.device().index(), self.scalar_type());
  std::tie(p_context_->syn_inputs_[0], std::ignore) = MatBwSize(
      graph,
      gradsum,
      grad_out,
      (other.dim() > 1) ? transpose.GetOutputs()[0] : memcpy.GetOutputs()[0],
      self.sizes(),
      std::move(p_context_->syn_inputs_[0]),
      (other.dim() > 1) ? std::move(transpose.GetSynOutputs()[0])
                        : std::move(memcpy.GetSynOutputs()[0]),
      is_output_persistent[0]);

  p_context_->syn_outputs_.emplace_back(std::move(gradsum.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(gradsum.GetOutputs()[0]));

  // grad_other = AD_matmul_bw_size(AD_mat_transpose(self), grad_output,
  // other_size)._grad_sum_to_size(other_size)
  TransposeOperator transpose1(self.device().index(), self.scalar_type());
  if (self.dim() > 1) {
    p_context_->syn_inputs_[1] = MatBwTranspose(
        graph, transpose1, self, std::move(p_context_->syn_inputs_[1]));
  } else {
    p_context_->syn_inputs_[1] = MatBwTranspose(
        graph, memcpy, self, std::move(p_context_->syn_inputs_[1]));
  }

  GradSumToSizeOperator gradsum1(self.device().index(), self.scalar_type());
  std::tie(std::ignore, p_context_->syn_inputs_[0]) = MatBwSize(
      graph,
      gradsum1,
      (self.dim() > 1) ? transpose1.GetOutputs()[0] : memcpy.GetOutputs()[0],
      grad_out,
      other.sizes(),
      (self.dim() > 1) ? std::move(transpose1.GetSynOutputs()[0])
                       : std::move(memcpy.GetSynOutputs()[0]),
      std::move(p_context_->syn_inputs_[0]),
      is_output_persistent[1]);

  p_context_->syn_outputs_.emplace_back(std::move(gradsum1.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(gradsum1.GetOutputs()[0]));
}

std::tuple<Tensor, Tensor> matmul_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& other) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = grad_output.scalar_type();
  std::string node_type =
      "matmul_backward" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = grad_output.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  habana::MatmulBackwardOperator Op(device_id);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(grad_output), IValue(self), IValue(other)};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{grad_output, self, other};
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output1 = at::empty(self.sizes(), self.options());
    auto output2 = at::empty(other.sizes(), other.options());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs({output1, output2});
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  PT_KERNEL_END;
  return std::make_tuple(out.at(0), out.at(1));
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::mm",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::MMOperator>(device_id);
            })
        .add(
            "aten::mv",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::MvOperator>(device_id);
            })
        .add(
            "aten::dot",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::DotOperator>(device_id);
            })
        .add(
            "aten::addmm",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::AddmmOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::bmm",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::BmmOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::matmul_backward",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::MatmulBackwardOperator>(
                  device_id);
            })
        .add("aten::matmul",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::MatMulOperator>(device_id);
        });
