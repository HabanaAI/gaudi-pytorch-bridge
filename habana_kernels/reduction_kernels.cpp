/**
 * Copyright (c) 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "habana_kernels/reduction_kernels.h"
#include <absl/container/fixed_array.h>
#include <perf_lib_layer_params.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/script.h>
#include <cstdint>
#include "backend/create_pt_tensor.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/lowering_util.h"
#include "habana_helpers/conversion.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;
using namespace habana;

namespace {

void allocate_reduction_result(
    Tensor& result,
    const Tensor& self,
    DimMask mask,
    bool keepdim,
    ScalarType dtype,
    bool is_result_persistent) {
  auto shape = DimVector(self.sizes());
  const auto shape_size = shape.size();
  HABANA_ASSERT(
      shape_size <= static_cast<size_t>(std::numeric_limits<int>::max()),
      "Shape size is too large for int conversion: ",
      shape_size);
  for (int dim = static_cast<int>(shape_size) - 1; dim >= 0; dim--) {
    const auto dim_size_t = static_cast<size_t>(dim);
    if (mask[dim_size_t]) {
      if (keepdim) {
        shape[dim_size_t] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }

  // Following code is required to convert Pytorch 0d tensor
  // to a 1d tensor. This is required because synapse_helpers
  // tensor_builder does not support 0d tensors
  if (shape.empty()) {
    shape.push_back(1);
  }

  if (result.defined()) {
    auto tht_result = result.unsafeGetTensorImpl();
    if (result.numel() || is_result_persistent)
      THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);
    else {
      THHTensor_resizeNd_nonpersistent(
          tht_result, shape.size(), shape.data(), nullptr);
    }
  } else {
    auto memory_format = self.suggest_memory_format();
    if (shape.size() < 4) {
      memory_format = at::MemoryFormat::Contiguous;
    }
    result = habana::createPTTensor(
        self,
        shape,
        self.options().dtype(dtype),
        memory_format,
        is_result_persistent);
  }
}

} // namespace

std::vector<int64_t> ReduceOperator::compute_output_shape(
    const at::Tensor& self,
    const IntArrayRef dim,
    const bool keepdim) {
  return LoweringUtil::ComputeOutputShape(self, dim, keepdim);
}

void ReduceOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor output = inputs[0].toTensor();
  Tensor self = inputs[1].toTensor();
  auto dim = inputs[2].toIntList();
  bool keepdim = inputs[3].toBool();
  auto dtype = inputs[4].toOptional<ScalarType>();

  absl::FixedArray<int64_t> data(dim.begin(), dim.end());
  IntArrayRef dim_arr(data.data(), dim.size());
  const auto ndim = self.dim();
  const auto mask = LoweringUtil::MakeDimMask(dim_arr, ndim);

  allocate_reduction_result(
      output,
      self,
      mask,
      keepdim,
      LoweringUtil::GetDtype(output, self, dtype, false),
      true);
  /*HABANA_ASSERT(
      output.scalar_type() == self.scalar_type(),
      "Habana reduction ops don't support casts yet");*/
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

InferOutputMetaRetType ReduceOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  // output tensor
  Tensor output = inputs[0].toTensor();
  Tensor self = inputs[1].toTensor();
  auto in_dim = inputs[2].toIntVector();
  bool keepdim = inputs[3].toBool();
  auto dtype = inputs[4].toOptional<ScalarType>();
  auto num_dims_to_reduce = in_dim.size();
  // wrap dims to positive values, sort dim list and remove any duplicates
  LoweringUtil::SortAndRemoveDuplicateDims(in_dim, self.dim());

  std::vector<int64_t> next_val{0, 1, 2, 3, 4};
  bool flatten_higher_dims = false;
  for (auto i = 0U; i < num_dims_to_reduce && num_dims_to_reduce > 1; i++) {
    if (in_dim[i] == next_val[i]) {
      flatten_higher_dims = true;
    } else {
      flatten_higher_dims = false;
      break;
    }
  }
  at::Tensor self_reshaped = self;
  if (flatten_higher_dims) {
    unsigned reshaped_in_dim_size = 1;
    std::vector<int64_t> reshaped_self_sizes;
    auto original_self_sizes = self.sizes().vec();

    auto flatten_size = std::accumulate(
        original_self_sizes.begin(),
        original_self_sizes.begin() +
            static_cast<ptrdiff_t>(num_dims_to_reduce),
        1,
        std::multiplies<int>());
    if (keepdim) {
      for (size_t i = 0; i < num_dims_to_reduce - 1; i++) {
        reshaped_self_sizes.emplace_back(1);
      }
    }
    reshaped_self_sizes.emplace_back(flatten_size);
    const auto self_dim = self.dim();
    HABANA_ASSERT(
        self_dim >= 0 && static_cast<size_t>(self_dim) >= num_dims_to_reduce,
        "Invalid tensor dimensions for reduction");
    for (size_t i = num_dims_to_reduce; i < static_cast<size_t>(self_dim);
         i++) {
      reshaped_self_sizes.emplace_back(original_self_sizes[i]);
    }
    // reshape to "reshaped-sizes" before reduction
    c10::IntArrayRef shape(
        reshaped_self_sizes.data(), reshaped_self_sizes.size());

    auto ReshapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, self.scalar_type());
    std::vector<c10::IValue> stack;
    stack.emplace_back(self);
    stack.emplace_back(shape);
    auto& reshape_out = out.call_InferOutputMeta(ReshapeOp, stack);
    self_reshaped = std::get<1>(reshape_out.GetOutputTensor(0));

    absl::FixedArray<int64_t> reshaped_in_dim_data(reshaped_in_dim_size);
    if (!keepdim) {
      reshaped_in_dim_data[0] = 0;
    } else {
      std::copy(
          in_dim.begin() + static_cast<ptrdiff_t>(num_dims_to_reduce - 1),
          in_dim.end(),
          reshaped_in_dim_data.data());
    }

    IntArrayRef reshaped_in_dim(
        reshaped_in_dim_data.data(), reshaped_in_dim_size);

    auto mask = LoweringUtil::MakeDimMask(reshaped_in_dim, self_reshaped.dim());
    allocate_reduction_result(
        output,
        self_reshaped,
        mask,
        keepdim,
        LoweringUtil::GetDtype(output, self_reshaped, dtype, false),
        false);
  } else {
    absl::FixedArray<int64_t> in_dim_copy(in_dim.begin(), in_dim.end());
    IntArrayRef in_dim_arr(in_dim_copy.data(), in_dim.size());
    auto mask = LoweringUtil::MakeDimMask(in_dim_arr, self.dim());
    allocate_reduction_result(
        output,
        self,
        mask,
        keepdim,
        LoweringUtil::GetDtype(output, self, dtype, false),
        false);
  }
  if (!keepdim) {
    auto shape_out = output.sizes().vec();
    auto out_metadata = TensorMetaData(
        shape_out,
        HabanaOperator::CalculateStrides(
            shape_out, self_reshaped.suggest_memory_format()),
        self_reshaped.scalar_type(),
        self_reshaped.suggest_memory_format());
    out.AddOutputTensor(out_metadata);

    auto ReshapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, self_reshaped.scalar_type());
    std::vector<c10::IValue> stack;
    stack.emplace_back(self_reshaped);
    stack.emplace_back(output.sizes());
    // reshape output
    out.call_InferOutputMeta(ReshapeOp, stack);
    // since reshape is directly realized at synapse guid level
    auto& reshape = out.GetKernel(out.GetKernelSize() - 1);
    reshape.RemoveOutput(0);
  }
  return out;
}

void ReduceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(
      inputs.size() == 5,
      "Incorrect size of inputs expected for reduction operator");
  HABANA_ASSERT(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for reduction operator");
  HABANA_ASSERT(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for reduction operator");
  HABANA_ASSERT(
      inputs[2].isIntList(),
      "Input arg3 expected to be IntList for reduction operator");
  HABANA_ASSERT(
      inputs[3].isBool(),
      "Input arg4 expected to be Bool for reduction operator");

  Tensor output = inputs[0].toTensor();
  Tensor self = inputs[1].toTensor();
  auto in_dim = inputs[2].toIntVector();
  bool keepdim = inputs[3].toBool();
  auto dtype = inputs[4].toOptional<ScalarType>();
  auto num_dims_to_reduce = in_dim.size();
  // wrap dims to positive values, sort dim list and remove any duplicates
  LoweringUtil::SortAndRemoveDuplicateDims(in_dim, self.dim());

  // check whether all dims in list are the higher "continuous" dimensions
  // if yes, "flatten" higher dims to a single unrolled-size dim.
  // Note-1 that this is an optimization to avoid any precision loss we may
  // get due to separate back 2 back reductions along single dimensions.
  // Note-2 cases such as [0,1,3] where there is in additional dim to reduce
  // in addition to continuous dims is not supported with flattening and falls
  // back to regular flow
  std::vector<int64_t> next_val{0, 1, 2, 3, 4};
  bool flatten_higher_dims = false;
  for (auto i = 0U; i < num_dims_to_reduce && num_dims_to_reduce > 1; i++) {
    if (in_dim[i] == next_val[i]) {
      flatten_higher_dims = true;
    } else {
      flatten_higher_dims = false;
      break;
    }
  }

  if (flatten_higher_dims) {
    // reshaped_sizes is used to hold appropriate dim sizes
    // reshaped_sizes is passed to Reshape operator stack as
    // an IntArrayRef variable.
    unsigned reshaped_in_dim_size = 1;
    std::vector<int64_t> reshaped_self_sizes;
    auto original_self_sizes = self.sizes().vec();

    auto flatten_size = std::accumulate(
        original_self_sizes.begin(),
        original_self_sizes.begin() +
            static_cast<ptrdiff_t>(num_dims_to_reduce),
        1,
        std::multiplies<int>());
    if (keepdim) {
      // we need to keep a size of '1' for upper dims, flattened value at last
      // pos of "dim array", and original sizes for lower dimensions
      // example: sizes [8,3,2,2] with dim=[0,1,2] and keepdim=true becomes
      // [1,1,48,2]
      for (size_t i = 0; i < num_dims_to_reduce - 1; i++) {
        reshaped_self_sizes.emplace_back(1);
      }
    }
    // else all upper dims sizes are flattened into a single dim at 0
    // example: sizes [8,3,2,2] with dim=[0,1,2] and keepdim=true becomes
    // [48,2]

    reshaped_self_sizes.emplace_back(flatten_size);
    const auto self_dim2 = self.dim();
    HABANA_ASSERT(
        self_dim2 >= 0 && static_cast<size_t>(self_dim2) >= num_dims_to_reduce,
        "Invalid tensor dimensions for reduction in AllocateAndAddSynapseNode");
    for (size_t i = num_dims_to_reduce; i < static_cast<size_t>(self_dim2);
         i++) {
      reshaped_self_sizes.emplace_back(original_self_sizes[i]);
    }
    // reshape to "reshaped-sizes" before reduction
    c10::IntArrayRef shape(
        reshaped_self_sizes.data(), reshaped_self_sizes.size());
    // Reshape operator to flatten higher dims to single dim.
    // The reshape node in the else part doesn't actually reshape, but is a pass
    // through. We assume that the reshape in the else part (when upper dims are
    // not merged), will be optimized out by GC
    auto ReshapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, self.scalar_type());
    ReshapeOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    // Build Params for the graph
    std::vector<c10::IValue> stack;
    stack.emplace_back(self);
    stack.emplace_back(shape);
    ReshapeOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

    auto self_reshaped = ReshapeOp->GetOutputs()[0];
    absl::FixedArray<int64_t> reshaped_in_dim_data(reshaped_in_dim_size);
    if (!keepdim) {
      reshaped_in_dim_data[0] = 0;
    } else {
      std::copy(
          in_dim.begin() + static_cast<ptrdiff_t>(num_dims_to_reduce - 1),
          in_dim.end(),
          reshaped_in_dim_data.data());
    }

    IntArrayRef reshaped_in_dim(
        reshaped_in_dim_data.data(), reshaped_in_dim_size);
    auto mask = LoweringUtil::MakeDimMask(reshaped_in_dim, self_reshaped.dim());
    allocate_reduction_result(
        output,
        self_reshaped,
        mask,
        keepdim,
        LoweringUtil::GetDtype(output, self_reshaped, dtype, false),
        output_metadata.at(0).persistent);
    /*HABANA_ASSERT(
        output.scalar_type() == self_reshaped.scalar_type(),
        "Habana reduction ops don't support casts yet");*/
    AllocateSynapseOutput(graph, output, output_metadata.at(0));

    std::tie(std::ignore, p_context_->syn_outputs_[0]) = CreateReductionGraph(
        graph,
        self_reshaped,
        output,
        std::move(ReshapeOp->GetSynOutputs()[0]),
        std::move(p_context_->syn_outputs_[0]),
        reshaped_in_dim,
        keepdim);
  } else {
    absl::FixedArray<int64_t> in_dim_copy(in_dim.begin(), in_dim.end());
    IntArrayRef in_dim_arr(in_dim_copy.data(), in_dim.size());
    auto mask = LoweringUtil::MakeDimMask(in_dim_arr, self.dim());
    allocate_reduction_result(
        output,
        self,
        mask,
        keepdim,
        LoweringUtil::GetDtype(output, self, dtype, false),
        output_metadata.at(0).persistent);
    /*HABANA_ASSERT(
        output.scalar_type() == self.scalar_type(),
        "Habana reduction ops don't support casts yet");*/
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
    std::tie(p_context_->syn_inputs_[0], p_context_->syn_outputs_[0]) =
        CreateReductionGraph(
            graph,
            self,
            output,
            std::move(p_context_->syn_inputs_[0]),
            std::move(p_context_->syn_outputs_[0]),
            in_dim_arr,
            keepdim);
  }
}

static std::vector<std::string> multi_output_reduce_ops = {
    "reduce_min_fwd",
    "reduce_max_fwd"};
int ReduceOperator::get_num_tpc_outputs() {
  for (size_t i = 0; i < multi_output_reduce_ops.size(); i++) {
    if (guid_.find(multi_output_reduce_ops[i]) != std::string::npos) {
      return 2;
    }
  }
  return 1;
}
std::tuple<synapse_helpers::tensor_or_ref, synapse_helpers::tensor_or_ref>
ReduceOperator::CreateReductionGraph(
    synapse_helpers::graph& graph,
    Tensor& pyt_tensor,
    const at::Tensor& output,
    synapse_helpers::tensor_or_ref syn_tensor_in,
    synapse_helpers::tensor_or_ref syn_tensor_out,
    IntArrayRef in_dim,
    bool keepdim) {
  // In the code below syn_helper_intermediate[0] holds the reshaped/original
  // input, syn_helper_intermediate[<last_index>] holds the final output and
  // all others in between holds intermediate output/net-stage-input in the
  // dim by dim reductions.
  std::vector<synapse_helpers::tensor_or_ref> syn_helper_intermediate;
  std::vector<int64_t> pyt_shape = pyt_tensor.sizes().vec();
  auto pyt_stride = pyt_tensor.strides().vec();
  ScalarType dtype = output.scalar_type();
  auto num_tpc_outputs = get_num_tpc_outputs();
  int first_input_pos = 1 - num_tpc_outputs;
  // add syn_input tensor
  synapse_helpers::tensor& synInput = syn_tensor_in;
  syn_helper_intermediate.emplace_back(synInput);
  // create syn_intermediate tensors of required shape
  const auto in_dim_size = in_dim.size();
  HABANA_ASSERT(
      in_dim_size <= static_cast<size_t>(std::numeric_limits<unsigned>::max()),
      "in_dim size is too large for unsigned conversion: ",
      in_dim_size);
  using namespace std::literals;
  const unsigned loopend = keepdim
      ? safe_convert<unsigned>(in_dim_size - 1, "loopend keepdim"sv)
      : safe_convert<unsigned>(in_dim_size, "loopend"sv);
  for (unsigned i = 0; i < loopend; i++) {
    const auto dim_index = in_dim[i];
    HABANA_ASSERT(
        dim_index >= 0, "Dimension index must be non-negative: ", dim_index);
    pyt_shape[safe_convert<size_t>(dim_index, "pyt_shape index"sv)] = 1;

    // Modify the stride accordingly after the shape change above
    pyt_stride[pyt_shape.size() - 1] = 1;
    for (size_t d = pyt_shape.size() - 1; d > 0; --d) {
      pyt_stride[d - 1] = pyt_stride[d] * pyt_shape[d];
    }

    c10::IntArrayRef shape(pyt_shape.data(), pyt_shape.size());
    syn_helper_intermediate.emplace_back(
        habana_helpers::create_tensor(
            shape,
            pyt_stride,
            graph,
            false,
            false,
            pyt_tensor.device().index(),
            dtype));
    if (num_tpc_outputs != 1) {
      // create second tensor for index
      syn_helper_intermediate.emplace_back(
          habana_helpers::create_tensor(
              shape,
              pyt_stride,
              graph,
              false,
              false,
              pyt_tensor.device().index(),
              c10::ScalarType::Int));
    }
  }
  // add syn_output tensor
  synapse_helpers::tensor& synOutput = syn_tensor_out;
  syn_helper_intermediate.emplace_back(synOutput);
  if (keepdim && num_tpc_outputs != 1) {
    // create second tensor for index
    syn_helper_intermediate.emplace_back(
        habana_helpers::create_tensor(
            output.sizes(),
            output.strides(),
            graph,
            false,
            false,
            pyt_tensor.device().index(),
            c10::ScalarType::Int));
  }
  /*
  i=0, o=1,2
  i=1, o=3,4
  i=3, o=5,6
  i=5, o=7,8
  i=7, o=9,10
  */
  // add reduction nodes corresponding to intermediate stages
  HABANA_ASSERT(
      num_tpc_outputs >= 0,
      "num_tpc_outputs must be non-negative: ",
      num_tpc_outputs);
  const auto num_tpc_outputs_u = static_cast<size_t>(num_tpc_outputs);
  const auto loop_limit = num_tpc_outputs * in_dim_size;
  for (size_t i = 0, j = 0; i < loop_limit; i += num_tpc_outputs_u, j++) {
    std::string node_type = this->guid_;
    ns_Reduction::Params params{};
    const auto tensor_dim = pyt_tensor.dim();
    const auto reduction_dim = tensor_dim - in_dim[j] - 1;
    params.reductionDimension =
        safe_convert<unsigned int>(reduction_dim, "reductionDimension"sv);
    const int offset_calc = static_cast<int>(i) + (i != 0) * first_input_pos;
    const auto input_index_offset =
        safe_convert<size_t>(offset_calc, "input_index_offset"sv);
    std::vector<synTensor> syn_in{
        syn_helper_intermediate[input_index_offset].ref().get()};
    std::vector<synTensor> syn_out{syn_helper_intermediate[i + 1].ref().get()};
    if (num_tpc_outputs != 1) {
      syn_out.emplace_back(syn_helper_intermediate[i + 2].ref().get());
    }
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &params,
        sizeof(params),
        std::move(node_type),
        nullptr,
        nullptr,
        nullptr,
        deterministic,
        getContextHints());
  }
  // if dim need not be kept add a final reshape to remove the "1" sized upper
  // dims
  if (!keepdim) {
    std::string node_type = "reshape";
    const auto input_index_offset = [&] {
      if (num_tpc_outputs > 1) {
        const auto num_tpc_outputs_u = safe_convert<size_t>(
            num_tpc_outputs, "num_tpc_outputs for reshape"sv);
        return num_tpc_outputs_u * in_dim_size - 1;
      } else {
        return in_dim_size;
      }
    }();
    std::vector<synTensor> syn_in{
        syn_helper_intermediate[input_index_offset].ref().get()};
    std::vector<synTensor> syn_out{
        syn_helper_intermediate
            [safe_convert<size_t>(
                 num_tpc_outputs, "num_tpc_outputs for syn_out"sv) *
                 in_dim_size +
             1]
                .ref()
                .get()};

    auto reshapeOp =
        make_operator<ReshapeOperator>(p_context_->device_id_, dtype);
    if (graph.is_dynamic_graph()) {
      reshapeOp->AllocateSynapseShapeTensor(graph, output);
      synapse_helpers::tensor& syn_shape = reshapeOp->GetSynInputs().back();
      syn_in.emplace_back(syn_shape.get());
    }

    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        nullptr,
        0,
        node_type,
        nullptr,
        nullptr,
        nullptr,
        deterministic,
        getContextHints());
  }
  return std::make_tuple(std::move(syn_tensor_in), std::move(syn_tensor_out));
}

InferOutputMetaRetType SumDimOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  if (inputs.size() == 4) {
    auto self = inputs[0].toTensor();
    auto dim = inputs[1].toIntVector();
    bool keepdim = inputs[2].toBool();

    // Check if dim = [], if yes, reduce input along all dims
    // dim = tuple(range(self.dim))
    if (dim.empty()) {
      for (int i = 0; i < self.dim(); ++i) {
        dim.push_back(i);
      }
      inputs[1] = dim;
    }
    // Remove duplicates in dim list
    LoweringUtil::SortAndRemoveDuplicateDims(dim, self.dim());
    // compute number of output dims
    const auto self_dim = self.dim();
    const auto dim_size = dim.size();
    HABANA_ASSERT(
        self_dim >= 0 && dim_size <= static_cast<size_t>(self_dim),
        "Invalid dimensions for output_dims calculation: self.dim=",
        self_dim,
        ", dim.size=",
        dim_size);
    const auto output_dims =
        self_dim - (!(keepdim) * static_cast<int64_t>(dim_size));
    // output follows input memory_format for all cases
    // except when output has less than 4 dims
    auto memory_format = self.suggest_memory_format();
    if (output_dims < 4) {
      memory_format = at::MemoryFormat::Contiguous;
    }
    Tensor output =
        habana::createPTTensor(self, {0}, self.options(), memory_format, false);
    inputs.insert(inputs.begin(), IValue(output));
  }
  return ReduceOperator::InferOutputMeta(inputs);
}
void SumDimOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(
      inputs.size() == 4,
      "Incorrect size of inputs expected for SumDim operator");
  HABANA_ASSERT(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for SumDim operator");
  HABANA_ASSERT(
      inputs[1].isIntList(),
      "Input arg2 expected to be IntList for SumDim operator");
  HABANA_ASSERT(
      inputs[2].isBool(), "Input arg3 expected to be Bool for SumDim operator");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toIntVector();
  bool keepdim = inputs[2].toBool();

  // Check if dim = [], if yes, reduce input along all dims
  // dim = tuple(range(self.dim))
  if (dim.empty()) {
    for (int i = 0; i < self.dim(); ++i) {
      dim.push_back(i);
    }
    inputs[1] = dim;
  }
  // Remove duplicates in dim list
  LoweringUtil::SortAndRemoveDuplicateDims(dim, self.dim());
  // compute number of output dims
  auto output_dims_signed =
      self.dim() - (!(keepdim) * static_cast<int64_t>(dim.size()));
  HABANA_ASSERT(output_dims_signed >= 0, "output_dims must be non-negative");
  auto output_dims = static_cast<size_t>(output_dims_signed);
  // output follows input memory_format for all cases
  // except when output has less than 4 dims
  auto memory_format = self.suggest_memory_format();
  if (output_dims < 4) {
    memory_format = at::MemoryFormat::Contiguous;
  }
  Tensor output = habana::createPTTensor(
      self,
      {0},
      self.options(),
      memory_format,
      output_metadata.at(0).persistent);
  inputs.insert(inputs.begin(), IValue(output));

  ReduceOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}

void SumDimOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor output;
  inputs.insert(inputs.begin(), IValue(output));
  ReduceOperator::SetPTOutputs(inputs);
}

InferOutputMetaRetType SumDimOutOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toIntList();
  auto output = inputs[4].toTensor();

  // Create a new container with all dims of input tensor, followed by creation
  // of a new reference to it. This is used in case "dim" provided is {}, which
  // implies that all dims need to be reduced.
  std::vector<int64_t> data;
  auto ndim = self.dim();
  for (int i = 0; i < ndim; i++) {
    data.push_back(i);
  }
  IntArrayRef dim_new(data);

  // Check if dim = {}, if yes, reduce input along all dims
  if (dim.empty()) {
    inputs[1] = IValue(dim_new);
  }

  // Move the output at begining
  inputs.insert(inputs.begin(), IValue(output));
  inputs.erase(inputs.end());
  return ReduceOperator::InferOutputMeta(inputs);
}

void SumDimOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(
      inputs.size() == 5,
      "Incorrect size of inputs expected for SumDimOut operator");
  HABANA_ASSERT(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for SumDimOut operator");
  HABANA_ASSERT(
      inputs[1].isIntList(),
      "Input arg2 expected to be IntList for SumDimOut operator");
  HABANA_ASSERT(
      inputs[2].isBool(),
      "Input arg3 expected to be Bool for SumDimOut operator");
  HABANA_ASSERT(
      inputs[4].isTensor(),
      "Input arg5 expected to be tensor for SumDimOut operator");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toIntList();
  auto output = inputs[4].toTensor();

  // Create a new container with all dims of input tensor, followed by creation
  // of a new reference to it. This is used in case "dim" provided is {}, which
  // implies that all dims need to be reduced.
  std::vector<int64_t> data;
  auto ndim = self.dim();
  for (int i = 0; i < ndim; i++) {
    data.push_back(i);
  }
  IntArrayRef dim_new(data);

  // Check if dim = {}, if yes, reduce input along all dims
  if (dim.empty()) {
    inputs[1] = IValue(dim_new);
  }

  // Move the output at begining
  inputs.insert(inputs.begin(), IValue(output));
  inputs.erase(inputs.end());
  ReduceOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}

void SumDimOutOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  ReduceOperator::SetPTOutputs(inputs);
}

InferOutputMetaRetType SumOperator::InferOutputMeta(torch::jit::Stack& inputs) {
  if (inputs.size() == 2) {
    Tensor self = inputs[0].toTensor();
    Tensor output = habana::createPTTensor(
        self, {0}, self.options(), at::MemoryFormat::Contiguous, false);

    auto ndim = self.dim();
    int64_t data[HABANA_DIM_MAX];
    for (int i = 0; i < ndim; i++) {
      data[i] = i;
    }
    HABANA_ASSERT(ndim >= 0, "ndim must be non-negative");
    IntArrayRef dim(data, static_cast<size_t>(ndim));
    bool keepdim = false;

    inputs.insert(inputs.begin(), IValue(output));
    inputs.insert(inputs.begin() + 2, IValue(dim));
    inputs.insert(inputs.begin() + 3, IValue(keepdim));
  }
  return ReduceOperator::InferOutputMeta(inputs);
}
void SumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(
      inputs.size() == 2, "Incorrect size of inputs expected for Sum operator");
  HABANA_ASSERT(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Sum operator");

  Tensor self = inputs[0].toTensor();
  Tensor output = habana::createPTTensor(
      self,
      {0},
      self.options(),
      at::MemoryFormat::Contiguous,
      output_metadata.at(0).persistent);

  auto ndim = self.dim();
  int64_t data[HABANA_DIM_MAX];
  for (int i = 0; i < ndim; i++) {
    data[i] = i;
  }
  HABANA_ASSERT(ndim >= 0, "ndim must be non-negative");
  IntArrayRef dim(data, static_cast<size_t>(ndim));
  bool keepdim = false;

  inputs.insert(inputs.begin(), IValue(output));
  inputs.insert(inputs.begin() + 2, IValue(dim));
  inputs.insert(inputs.begin() + 3, IValue(keepdim));

  ReduceOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}

void SumOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  Tensor output;
  auto ndim = self.dim();
  int64_t data[HABANA_DIM_MAX];
  for (int i = 0; i < ndim; i++) {
    data[i] = i;
  }
  HABANA_ASSERT(ndim >= 0, "ndim must be non-negative");
  IntArrayRef dim(data, static_cast<size_t>(ndim));
  bool keepdim = false;

  inputs.insert(inputs.begin(), IValue(output));
  inputs.insert(inputs.begin() + 2, IValue(dim));
  inputs.insert(inputs.begin() + 3, IValue(keepdim));
  ReduceOperator::SetPTOutputs(inputs);
}

InferOutputMetaRetType MeanOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  if (inputs.size() == 2) {
    Tensor self = inputs[0].toTensor();
    Tensor output = habana::createPTTensor(
        self, {0}, self.options(), at::MemoryFormat::Contiguous, false);

    std::vector<int64_t> data;
    auto ndim = self.dim();
    for (int i = 0; i < ndim; i++) {
      data.push_back(i);
    }

    IntArrayRef dim(data);

    bool keepdim = false;
    inputs.insert(inputs.begin(), IValue(output));
    inputs.insert(inputs.begin() + 2, IValue(dim));
    inputs.insert(inputs.begin() + 3, IValue(keepdim));
  }
  return ReduceOperator::InferOutputMeta(inputs);
}
void MeanOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(
      inputs.size() == 2,
      "Incorrect size of inputs expected for Mean operator");
  HABANA_ASSERT(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Mean operator");

  Tensor self = inputs[0].toTensor();

  auto& mdata = output_metadata.at(0);
  Tensor output;
  if (!graph.is_dry_run() && mdata.allocated_tensor.has_value()) {
    output = mdata.allocated_tensor.value();
  } else {
    output = habana::createPTTensor(
        self,
        {0},
        self.options(),
        at::MemoryFormat::Contiguous,
        output_metadata.at(0).persistent);
  }

  std::vector<int64_t> data;
  auto ndim = self.dim();
  for (int i = 0; i < ndim; i++) {
    data.push_back(i);
  }

  IntArrayRef dim(data);

  bool keepdim = false;
  inputs.insert(inputs.begin(), IValue(output));
  inputs.insert(inputs.begin() + 2, IValue(dim));
  inputs.insert(inputs.begin() + 3, IValue(keepdim));

  ReduceOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}

void MeanOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor self = inputs[0].toTensor();
  Tensor output;
  auto ndim = self.dim();
  int64_t data[HABANA_DIM_MAX];
  for (int i = 0; i < ndim; i++) {
    data[i] = i;
  }
  HABANA_ASSERT(ndim >= 0, "ndim must be non-negative");
  IntArrayRef dim(data, static_cast<size_t>(ndim));
  bool keepdim = false;

  inputs.insert(inputs.begin(), IValue(output));
  inputs.insert(inputs.begin() + 2, IValue(dim));
  inputs.insert(inputs.begin() + 3, IValue(keepdim));
  ReduceOperator::SetPTOutputs(inputs);
}

void ReduceMultiOutputOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(
      inputs.size() == 4,
      "Incorrect size of inputs expected for MaxDimOperator");
  HABANA_ASSERT(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for MaxDimOperator");

  Tensor self = inputs[0].toTensor();
  Tensor output = habana::createPTTensor(
      self,
      compute_output_shape(
          self, inputs[1].toIntList().vec(), inputs[2].toBool()), //{},
      self.options(),
      at::MemoryFormat::Contiguous,
      output_metadata.at(0).persistent);
  inputs.insert(inputs.begin(), IValue(output));

  ReduceOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}
