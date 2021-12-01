/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/repeat.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/topk_kernels.h"
#include "synapse_helpers/recipe.h"

using namespace torch;

using namespace habana;
// ensure we get good values and indices for topk
inline void _allocate_or_resize_output_with_indices(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    int64_t k,
    bool values_persistent,
    bool indices_persistent) {
  auto result_sizes = self.sizes().vec();
  if (result_sizes.size() > 0) {
    result_sizes[dim] = k;
  }
  if (values.defined()) {
    TORCH_CHECK(
        self.options().type_equal(values.options()),
        "output values must be of same type as input");
    auto tht_values = values.unsafeGetTensorImpl();
    if (values.numel() || values_persistent)
      THHTensor_resizeNd(tht_values, self.dim(), result_sizes.data(), nullptr);
    else {
      THHTensor_resizeNd_nonpersistent(
          tht_values, self.dim(), result_sizes.data(), nullptr);
    }
  } else {
    values = at::empty(result_sizes, self.options());
  }
  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == c10::ScalarType::Int,
        "output indices must be of scalar type Int");
    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    auto tht_indices = indices.unsafeGetTensorImpl();
    if (indices.numel() || indices_persistent)
      THHTensor_resizeNd(tht_indices, self.dim(), result_sizes.data(), nullptr);
    else {
      THHTensor_resizeNd_nonpersistent(
          tht_indices, self.dim(), result_sizes.data(), nullptr);
    }
  } else {
    indices =
        at::empty(result_sizes, self.options().dtype(c10::ScalarType::Int));
  }
}

void TopkOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs expected for topk operator");

  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 expected to be tensor for topkout operator");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isInt(),
      "Input arg1 expected to be of type Int or Tensor for topkout operator");
  TORCH_CHECK(
      inputs[2].isInt(),
      "Input arg2 expected to be of type Int for topkout operator");
  TORCH_CHECK(
      inputs[3].isBool(),
      "Input arg3 expected to be of type Bool for topkout operator");
  TORCH_CHECK(
      inputs[4].isBool(),
      "Input arg4 expected to be of type Bool for topkout operator");
  TORCH_CHECK(
      inputs[5].isTensor(),
      "Input arg5 expected to be tensor for topkout operator");
  TORCH_CHECK(
      inputs[6].isTensor(),
      "Input arg6 expected to be tensor for topkout operator");
  TORCH_CHECK(
      is_output_persistent.size() == 2,
      "TopkOutOperator: #is_output_persistent should be 2");

  auto self = inputs[0].toTensor();
  int64_t dim_ = inputs[2].toInt();
  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto values = inputs[5].toTensor();
  auto indices = inputs[6].toTensor();

  int64_t k;
  // Get k value
  if (inputs[1].isTensor()) {
    TORCH_CHECK(p_context_->syn_inputs_.back().ref().is_shape_tensor());
    TORCH_CHECK(p_context_->syn_inputs_.size() == 2);
    Tensor k_tensor = inputs[1].toTensor();
    k = k_tensor.sizes().vec().at(
        0); // Get the first element which holds the dynamic value of k
  } else {
    k = inputs[1].toInt();
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      Tensor k_tensor = habana_helpers::createPTTensor(
          self, k, self.options(), self.suggest_memory_format(), false);
      AllocateSynapseShapeTensor(graph, k_tensor);
    }
  }

  /*
     To support dynamic shape, the TPC kernel inputs needs to be {values_tensor,
     indices_tensor, null, k_tensor}. The following code create 3 additional ops
     to create the indices tensor: arrnage op -> reshape op -> repeat op
  */
  synapse_helpers::tensor& syn_in_self = p_context_->syn_inputs_[0];
  std::vector<synTensor> syn_inputs{syn_in_self.get()};

  if (inputs[3].isTensor() || graph.is_dynamic_graph()) {
    torch::jit::Stack temp_stack;
    const auto input_shape = self.sizes();
    const int start = 0;
    const int limit = input_shape[dim];
    const int step = 1;

    // Add arange op - the input tensor for the arange op is also the output
    // tensor
    auto arange_input_output_scalar_type = c10::ScalarType::Int;
    int input_output_depth = ArangeOperator::GetOutputSize(start, limit, step);
    std::vector<int64_t> input_output_sizes_vec{input_output_depth};
    IntArrayRef input_output_shape(
        input_output_sizes_vec.data(), input_output_sizes_vec.size());
    auto arangeInputOutput = habana_helpers::createPTTensor(
        self,
        input_output_shape,
        self.options(),
        self.suggest_memory_format(),
        arange_input_output_scalar_type,
        false);
    auto arangeOp = make_operator<ArangeOperator>(
        this->p_context_->device_id_, arange_input_output_scalar_type);
    arangeOp->AllocateSynapseInput(graph, arangeInputOutput, false);
    temp_stack = {
        IValue(start), IValue(limit), IValue(step), IValue(arangeInputOutput)};
    arangeOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
    temp_stack.clear();

    // Add reshape op
    auto reshaped_shape = std::vector<int64_t>(self.ndimension(), 1);
    reshaped_shape[dim] = limit;
    auto reshapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, arange_input_output_scalar_type);
    temp_stack = {IValue(arangeOp->GetOutputs()[0]), IValue(reshaped_shape)};
    reshapeOp->SetSynapseInput(arangeOp->GetSynOutputs()[0]);
    reshapeOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
    temp_stack.clear();

    // Add repeat op
    auto repeatOp = make_operator<RepeatOperator>(
        this->p_context_->device_id_, arange_input_output_scalar_type);
    std::vector<int64_t> repeats = input_shape.vec();
    repeats[dim] = 1;
    temp_stack = {IValue(reshapeOp->GetOutputs()[0]), IValue(repeats)};
    repeatOp->SetSynapseInput(reshapeOp->GetSynOutputs()[0]);
    repeatOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
    temp_stack.clear();

    // Add relevant syn inputs to support dynamic shape
    synapse_helpers::tensor& syn_in_tensor_k = p_context_->syn_inputs_[1];
    synapse_helpers::tensor& syn_in_indices = repeatOp->GetSynOutputs()[0];
    syn_inputs.emplace_back(syn_in_indices.get());
    syn_inputs.emplace_back(nullptr);
    syn_inputs.emplace_back(syn_in_tensor_k.get());
  }

  bool largest = inputs[3].toBool();
  bool sorted = inputs[4].toBool();

  /*
   * BFloat16 is currently not supported. Look at the following jira for more
   * details https://jira.habana-labs.com/browse/SW-37999
   */
  TORCH_CHECK(
      !(self.dtype() == c10::ScalarType::BFloat16),
      "BFloat16 is not supported");

  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");
  // TPC doen't support unsorted or ascending order - but that applies only for
  // tensors with more than 1 element
  if (self.numel() > 1) {
    TORCH_CHECK(sorted == true, "unsorted output not supported")
  }

  _allocate_or_resize_output_with_indices(
      values,
      indices,
      self,
      dim,
      k,
      is_output_persistent[0],
      is_output_persistent[1]);
  indices_persistent = is_output_persistent[0];
  values_persistent = is_output_persistent[1];

  synBeamParams params;
  params.bsw = k;
  params.axis = self.dim() - dim - 1;
  params.bottomK = !largest;

  std::vector<at::Tensor> outputs{values, indices};
  AllocateSynapseOutputs(graph, outputs, is_output_persistent, {true, true});

  synapse_helpers::tensor& syn_out0 = p_context_->syn_outputs_[0];
  synapse_helpers::tensor& syn_out1 = p_context_->syn_outputs_[1];
  std::vector<synTensor> syn_outputs{syn_out0.get(), syn_out1.get()};

  // add topk node
  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      &params,
      sizeof(params),
      std::move(guid_));
}

void TopkOutOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  int64_t k = inputs[1].toInt();
  int64_t dim_ = inputs[2].toInt();
  auto values = inputs[5].toTensor();
  auto indices = inputs[6].toTensor();

  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  _allocate_or_resize_output_with_indices(
      values, indices, self, dim, k, true, true);
  std::vector<at::Tensor> v{values, indices};
  HabanaOperator::SetPTOutputs(v);
}

std::tuple<Tensor&, Tensor&> topk_out_hpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  PT_KERNEL_BEGIN;

  std::string node_type = "topk";

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<c10::IValue> stack = {
      IValue(self),
      IValue(k),
      IValue(dim_),
      IValue(largest),
      IValue(sorted),
      IValue(values),
      IValue(indices)};
  TopkOutOperator Op(device_id, node_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  PT_KERNEL_END;
  return std::forward_as_tuple(out.at(0), out.at(1));
}

void TopkOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for topk operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for topk operator");
  TORCH_CHECK(
      is_output_persistent.size() == 2,
      "TopkOperator: #is_output_persistent should be 2");

  Tensor self = inputs[0].toTensor();
  auto values = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent[0]);
  auto indices = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      is_output_persistent[1]);
  inputs.push_back(IValue(values));
  inputs.push_back(IValue(indices));

  TopkOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void TopkOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  auto values = habana_helpers::createPTTensor(
      self, {0}, self.options(), self.suggest_memory_format(), true);
  auto indices = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);

  inputs.push_back(IValue(values));
  inputs.push_back(IValue(indices));

  TopkOutOperator::SetPTOutputs(inputs);
}
std::tuple<Tensor, Tensor> topk_hpu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  PT_KERNEL_BEGIN;

  std::string node_type = "topk";

  // Create the operator
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(k), IValue(dim), IValue(largest), IValue(sorted)};
  TopkOperator Op(device_id, node_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  PT_KERNEL_END;
  return std::forward_as_tuple(out.at(0), out.at(1));
}

void SortOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for sort operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for sort operator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input arg2 expected to be of type Int for sort operator");
  TORCH_CHECK(
      inputs[2].isBool(),
      "Input arg3 expected to be of type Bool for sort operator");
  TORCH_CHECK(
      is_output_persistent.size() == 2,
      "SortOperator: #is_output_persistent should be 2");

  Tensor self = inputs[0].toTensor();
  int64_t dim_ = inputs[1].toInt();
  bool sorted = true; // topk supports only sorted output

  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  inputs.insert(inputs.begin() + 1, IValue(self.size(dim)));
  inputs.emplace_back(IValue(sorted));

  auto values = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent[0]);
  auto indices = habana_helpers::createPTTensor(
      self,
      {0},
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      is_output_persistent[1]);

  inputs.push_back(IValue(values));
  inputs.push_back(IValue(indices));

  TopkOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void SortOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  int64_t dim_ = inputs[1].toInt();
  bool sorted = true; // topk supports only sorted output

  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  inputs.insert(inputs.begin() + 1, IValue(self.size(dim)));
  inputs.emplace_back(IValue(sorted));

  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(c10::ScalarType::Int));

  inputs.push_back(IValue(values));
  inputs.push_back(IValue(indices));

  TopkOutOperator::SetPTOutputs(inputs);
}

/*************************************************************************
 * @brief Kernel implementation for sort OP
 *        out_sorted, out_indices = torch.sort(self, dim, descending)
 * @param [out] sorted - output tensor, 1-4D, FP32
 * @param [out] indices - output tensor, 1-4D, I32
 * @param [in] self - input tensor, 1-4D, FP32
 * @param [in] dim - along which dimension to sort, int64_t, default = -1
 * @param [in] descending - sorting order (ascending or descending), bool,
 *default = false
 ************************************************************************/
std::tuple<Tensor, Tensor> sort_hpu(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  PT_KERNEL_BEGIN;

  std::string node_type = "topk";

  // Create the operator
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(descending)};
  SortOperator Op(device_id, node_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  PT_KERNEL_END;
  return std::forward_as_tuple(out.at(0), out.at(1));
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::topk",
            [](const int device_id, c10::ScalarType node_type) {
              static_cast<void>(node_type);
              return std::make_shared<habana::TopkOperator>(device_id, "topk");
            })
        .add(
            "aten::topk.values",
            [](const int device_id, c10::ScalarType node_type) {
              static_cast<void>(node_type);
              return std::make_shared<habana::TopkOutOperator>(
                  device_id, "topk");
            })
        .add(
            "hpu::topk",
            [](const int device_id, c10::ScalarType node_type) {
              static_cast<void>(node_type);
              return std::make_shared<habana::TopkOperator>(device_id, "topk");
            })
        .add(
            "hpu::topk.values",
            [](const int device_id, c10::ScalarType node_type) {
              static_cast<void>(node_type);
              return std::make_shared<habana::TopkOutOperator>(
                  device_id, "topk");
            });
