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
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/nonzero_kernel.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;
using namespace habana;

void NonZeroOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for NonZero operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 expected to be tensor for NonZero operator");

  auto self = inputs[0].toTensor();
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  int elements = self.numel();
  auto output_shape = DimVector{elements, dimensions};
  auto shape_tensor_shape = DimVector{5};
  // Create PT output stage 2
  auto cordinates_of_true = habana_helpers::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);
  auto shape_tensor = habana_helpers::createPTTensor(
      self,
      shape_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);

  std::vector<at::Tensor> outputs{cordinates_of_true, shape_tensor};
  HabanaOperator::SetPTOutputs(outputs);
}

void NonZeroOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for NonZero operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 expected to be tensor for NonZero operator");
  TORCH_CHECK(
      output_metadata.size() == 2,
      "output_metadata expected to be vector of size 2");

  auto self = inputs[0].toTensor();

  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  int elements = self.numel();
  auto output_shape = DimVector{elements, dimensions};
  auto shape_tensor_shape = DimVector{5};
  auto cordinates_of_true = habana_helpers::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(0).persistent);
  auto shape_tensor = habana_helpers::createPTTensor(
      self,
      shape_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(1).persistent);
  // shape_tensor is of type UINT32 not supported by ScalarType, use synDataType
  synDataType synType = syn_type_uint32;
  AllocateSynapseOutput(graph, cordinates_of_true, output_metadata.at(0));
  AllocateSynapseOutput(
      graph,
      shape_tensor,
      synType,
      output_metadata.at(1),
      graph.is_dynamic_graph() ? true : false);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/*************************************************************************
 * @brief Kernel implementation for torch.nonzero operator
 * @param self - Input tensor
 ************************************************************************/
Tensor nonzero_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();

  // Output required of type Int64
  if (self.numel() == 0) {
    auto shape = DimVector{0, dimensions};
    auto output = habana_helpers::createPTTensor(
        self,
        shape,
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Long,
        true);
    PT_KERNEL_END;
    return output;
  }
  Tensor self_in = self;
  // Long not supported for lt and gt operator
  // casting to int
  if (scalar_type == ScalarType::Long) {
    self_in = habana_helpers::cast_tensor_to_integer(self);
  }
  std::string node_type =
      "non_zero_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // Create the operator
  NonZeroOperator Op(device_id, self_in.scalar_type());
  std::vector<at::Tensor> pt_inputs{self_in};
  std::vector<c10::IValue> stack = {IValue(self_in)};

  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    OutputMetaDataVector output_metadata(2);
    output_metadata.at(0).persistent = true;
    output_metadata.at(1).persistent = true;
    Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");
  // Slice the output to select only relevant data
  // The shape tensor = [Dimension_detail, num_relevantElement, N/U, N/U, N/U]
  // Since output tensor[elements,Dimension] already has correct dimension
  // we need to slice along row to get relevant elements to outputs
  // i.e slice the output from where_stage_2 of shape (elementsxDimensions)
  // to get correct output (relevantElementsxDimensions)
  auto end = out.at(1)[1].item<int64_t>();
  if (end == 0) {
    out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides(
        {0, dimensions}, {1, 1});
    PT_KERNEL_END;
    return out.at(0);
  }
  auto result = out.at(0).slice(0, 0, end, 1);
  // Remove this cast once index_put_ implementation using scatter_nd is
  // available
  auto output = habana_helpers::cast_tensor_to_long(result);
  PT_KERNEL_END;
  return output;
}

static auto& KernelRegistry =
    habana::KernelRegistry().add("hpu::nonzero", KERNEL_FN(NonZeroOperator));
