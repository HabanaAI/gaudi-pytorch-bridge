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
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
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
      "Input arg1 expected to be tensor for topk operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for topk operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for topk operator");
  TORCH_CHECK(
      inputs[3].isInt(),
      "Input arg4 expected to be of type Int for topk operator");
  TORCH_CHECK(
      inputs[4].isInt(),
      "Input arg5 expected to be of type Int for topk operator");
  TORCH_CHECK(
      inputs[5].isBool(),
      "Input arg6 expected to be of type Bool for topk operator");
  TORCH_CHECK(
      inputs[6].isBool(),
      "Input arg7 expected to be of type Bool for topk operator");
  TORCH_CHECK(
      is_output_persistent.size() == 2,
      "TopkOutOperator: #is_output_persistent should be 2");

  auto values = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto self = inputs[2].toTensor();
  int64_t k = inputs[3].toInt();
  int64_t dim_ = inputs[4].toInt();
  bool largest = inputs[5].toBool();
  bool sorted = inputs[6].toBool();

  /*
   * BFloat16 is currently not supported. Look at the following jira for more
   * details https://jira.habana-labs.com/browse/SW-37999
   */
  TORCH_CHECK(
      !(self.dtype() == c10::ScalarType::BFloat16),
      "BFloat16 is not supported");

  int64_t dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

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

  p_context_->params_.emplace<synBeamParams>(params);
  p_context_->params_size_ = sizeof(params);

  std::vector<at::Tensor> outputs{values, indices};
  AllocateSynapseOutputs(graph, outputs, is_output_persistent, {true, true});
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void TopkOutOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto values = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto self = inputs[2].toTensor();
  int64_t k = inputs[3].toInt();
  int64_t dim_ = inputs[4].toInt();

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
  std::vector<c10::IValue> stack = {IValue(values),
                                    IValue(indices),
                                    IValue(self),
                                    IValue(k),
                                    IValue(dim_),
                                    IValue(largest),
                                    IValue(sorted)};
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
  inputs.insert(inputs.begin(), IValue(indices));
  inputs.insert(inputs.begin(), IValue(values));

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

  inputs.insert(inputs.begin(), IValue(indices));
  inputs.insert(inputs.begin(), IValue(values));

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

  inputs.insert(inputs.begin(), IValue(indices));
  inputs.insert(inputs.begin(), IValue(values));

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

  inputs.insert(inputs.begin(), IValue(indices));
  inputs.insert(inputs.begin(), IValue(values));

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

static auto& KernelRegistry = habana::KernelRegistry().add(
    "aten::topk",
    [](const int device_id, c10::ScalarType node_type) {
      static_cast<void>(node_type);
      return std::make_shared<habana::TopkOperator>(device_id, "topk");
    });
