/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "hpu_ops/backend/reduction_template.h"
#include "backend/helpers/lowering_util.h"
#include "habana_kernels/kernel_utils.h"
#include "hpu_ops/common/reduction_template.h"

namespace habana {
void ReductionBackendTemplate::SetReductionVarsIndices(
    at::optional<uint8_t> dim_index,
    at::optional<uint8_t> keepdim_index,
    at::optional<uint8_t> dtype_index) {
  m_dim_index = dim_index;
  m_keepdim_index = keepdim_index;
  m_dtype_index = dtype_index;
}

ns_Reduction::ParamsV2 FillReductionParams(
    int64_t ndims,
    c10::IntArrayRef dims,
    bool keepdim) {
  ns_Reduction::ParamsV2 params;
  params.keepDim = keepdim;

  params.reductionDimensionMask = 0;
  for (auto&& dim : dims) {
    auto d = c10::maybe_wrap_dim(dim, ndims);
    if ((d >= 0) && (d < ndims)) {
      params.reductionDimensionMask |= (1 << (ndims - d - 1));
    }
  }

  return params;
}

void ReductionBackendTemplate::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  synTensor input = syn_in(0);

  // Extract dtype and cast self to the supplied dtype
  auto dtype = get_dtype(stack, m_dtype_index);

  auto cast = HandleReductionDtype(this, graph, self, input, dtype);
  if (cast.has_value()) {
    input = cast.value().get();
  }

  // Extract dims
  auto dims = get_dims(stack, m_dim_index);

  // Extract keepdim
  bool keepdim = get_keepdim(stack, m_keepdim_index);

  auto guid_name = GetGuid();

  if ((guid_name.find("reduce_") != std::string::npos) &&
      (guid_name.find("_multi_dim") != std::string::npos)) {
    // "========= Multi dim CGUID case ===================="
    auto shape = ReductionOutputShape(self, dims, keepdim)[0];
    int ndims = self.dim();

    NodeAttr::NodeOutputAttr reduction_node_output_attr = {
        shape, ScalarType(), 0};

    auto params = FillReductionParams(ndims, dims, keepdim);
    auto result = OpBackend::BuildNode(
        this,
        graph,
        {guid_name,
         {std::move(input)},
         {reduction_node_output_attr},
         &params,
         sizeof(params)});
    syn_out(0) = std::move(result[0]);
  } else {
    auto shape = ComputeOutputShapes(stack).empty()
        ? ReductionOutputShape(self, dims, keepdim)[0]
        : ComputeOutputShapes(stack)[0];

    // [SW-181253] - reduce_sum doesn't support Long dtype correctly, so the
    // explicit cast from int32 -> int64 is needed. When Long dtype support is
    // enabled and the input has integral dtype, it is expected that the output
    // tensor: https://github.com/pytorch/pytorch/issues/115832
    const bool shouldCastToLong = common::IsInt64Supported() &&
        at::isIntegralType(ScalarType(), true) && !dtype.has_value();
    std::vector<NodeAttr::NodeOutputAttr> output_attrs = shouldCastToLong
        ? std::vector<NodeAttr::NodeOutputAttr>{{shape, ScalarType()}}
        : std::vector<NodeAttr::NodeOutputAttr>{{shape, ScalarType(), 0}};

    auto result = HandleReductionDimAndKeepdim(
        this, graph, self, {input}, dims, keepdim, GetGuid(), output_attrs);

    if (shouldCastToLong) {
      auto casted = OpBackend::BuildCast(
          this, graph, result[0].get(), shape, torch::kInt, torch::kLong, 0);

      syn_out(0) = std::move(casted);
    } else {
      syn_out(0) = std::move(result[0]);
    }
  }
}

inline bool reduction_support_f32(const std::string& guid) {
  return guid.find("reduce_prod_multi_dim_fwd") != std::string::npos or guid.find("reduce_mean_multi_dim_fwd") != std::string::npos;
}

inline bool reduction_support_i32(const std::string& guid) {
  return guid.find("reduce_sum_multi_dim") != std::string::npos;
}

// Returns the input after cast to the supplied dtype. If dtype is none or if
// dtype is same as input's dtype, returns nullopt.
c10::optional<synapse_helpers::tensor> HandleReductionDtype(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synTensor syn_in,
    at::optional<at::ScalarType> dtype) {
  auto dtype_val = dtype.value_or(self.scalar_type());
  std::string guid = op->GetGuid();
  if (at::isIntegralType(self.scalar_type(), true) and reduction_support_i32(guid)) {
    if (dtype_val != at::kFloat) { dtype_val = at::kInt;}
  } else if (
      reduction_support_f32(guid) and at::isIntegralType(self.scalar_type(), true)) {
    dtype_val = at::kFloat;
  } else {
    return c10::nullopt;
    // do nothing
  }
  op->SetGuid(update_guid_dtype(guid, dtype_val));
  if (habana_helpers::getInternalDtype(dtype_val) ==
      habana_helpers::getInternalDtype(self.scalar_type())) {
    return c10::nullopt;
  }

  op->SetScalarType(dtype_val);

  return OpBackend::BuildCast(
      op, graph, syn_in, self.sizes(), self.scalar_type(), dtype_val);
}

void ProcessDim(std::vector<int64_t>& dims, const int& ndims) {
  // When dim=[], reduce all dimensions based on keepdim value
  if (0 == dims.size()) {
    for (int i = 0; i < ndims; ++i) {
      dims.push_back(i);
    }
  }
  LoweringUtil::SortAndRemoveDuplicateDims(dims, ndims);
}

bool CheckDims(const std::vector<int64_t>& dims) {
  auto num_dims_to_reduce = dims.size();
  bool flatten_higher_dims = false;
  for (auto i = 0u; i < num_dims_to_reduce && num_dims_to_reduce > 1; i++) {
    if (dims[i] == i) {
      flatten_higher_dims = true;
    } else {
      flatten_higher_dims = false;
      break;
    }
  }
  return flatten_higher_dims;
}

void CombineDims(
    const at::Tensor& self,
    std::vector<int64_t>& dim,
    const bool keepdim,
    std::vector<int64_t>& reshaped_self_sizes) {
  auto dim_size = dim.size();
  std::vector<int64_t> self_shape{self.sizes().vec()};
  auto flatten_size = std::accumulate(
      self_shape.begin(),
      self_shape.begin() + dim_size,
      (int64_t)1,
      std::multiplies<int64_t>());
  if (keepdim) {
    // we need to keep a size of '1' for upper dims, flattened value at last
    // pos of "dim array", and original sizes for lower dimensions
    // example: sizes [8,3,2,2] with dim=[0,1,2] and keepdim=true becomes
    // [1,1,48,2]
    for (unsigned i = 0; i < dim_size - 1; i++) {
      reshaped_self_sizes.emplace_back(1);
    }
  }
  // else all upper dims sizes are flattened into a single dim at 0
  // example: sizes [8,3,2,2] with dim=[0,1,2] and keepdim=true becomes
  // [48,2]
  reshaped_self_sizes.emplace_back(flatten_size);
  for (unsigned i = dim_size; i < self.dim(); i++) {
    reshaped_self_sizes.emplace_back(self_shape[i]);
  }
  std::vector<int64_t> reshaped_in_dim;
  if (!keepdim) {
    reshaped_in_dim.push_back(0);
  } else {
    reshaped_in_dim.push_back(dim[dim.size() - 1]);
  }
  dim = reshaped_in_dim;
}

static synapse_helpers::tensor FlattenInput(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor syn_in,
    const std::vector<int64_t>& reshaped_self_sizes) {
  return OpBackend::BuildReshape(
      op, graph, syn_in, reshaped_self_sizes, op->ScalarType());
}

static void GuidOutCount(
    OpBackend* op,
    const std::string& guid,
    const std::vector<int64_t>& retain_ten_shape,
    int& num_tpc_outputs,
    std::vector<NodeAttr::NodeOutputAttr>& output_attr,
    size_t num_outputs_required) {
  // TPC guids which returns two outputs
  static std::vector<std::string> multi_output_reduce_ops = {
      "reduce_min_fwd",
      "reduce_max_fwd",
      "reduce_Lp_fwd",
      "reduce_log_sum_exp_fwd",
      "reduce_log_sum_fwd"};
  std::vector<std::string>
      multi_output_reduce_ops_with_retain_tensor_int_dtype = {
          "reduce_min_fwd", "reduce_max_fwd"};

  bool retain_tensor_int = false;
  for (size_t i = 0;
       i < multi_output_reduce_ops_with_retain_tensor_int_dtype.size();
       i++) {
    if (guid.find(multi_output_reduce_ops_with_retain_tensor_int_dtype[i]) !=
        std::string::npos) {
      retain_tensor_int = true;
    }
  }

  for (size_t i = 0; i < multi_output_reduce_ops.size(); i++) {
    if (guid.find(multi_output_reduce_ops[i]) != std::string::npos) {
      num_tpc_outputs = 2;
      // when caller needs only one output but TPC returns two output
      if (num_outputs_required == 1) {
        output_attr.push_back(
            {retain_ten_shape,
             (retain_tensor_int == true) ? c10::ScalarType::Int
                                         : op->ScalarType()});
      }
      break;
    }
  }
}

std::vector<synapse_helpers::tensor> HandleReductionDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    std::vector<synTensor> inputs,
    const at::IntArrayRef dims,
    bool keepdim,
    const std::string& guid,
    std::vector<NodeAttr::NodeOutputAttr> output_attr) {
  return HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      std::move(inputs),
      dims,
      keepdim,
      guid,
      output_attr,
      [](const int ndim, size_t& size, int64_t index, c10::optional<at::Scalar>)
          -> std::shared_ptr<void> {
        PARAMS_STUB(ns_Reduction::Params);
        auto reduction_dim = ndim - 1 - index;
        params->reductionDimension = reduction_dim;
        return params;
      });
}

std::vector<int64_t> CalculateReductionMultiDimAndKeepdimOutputSize(
    const std::vector<int64_t>& inputSize,
    const std::vector<int64_t>& dimsToReduce,
    bool keepDim) {
  if (keepDim) {
    std::vector<int64_t> outputSize = inputSize;
    for (const int64_t dim : dimsToReduce) {
      outputSize[dim] = 1;
    }
    return outputSize;
  } else {
    const size_t numOfDimsLeft = inputSize.size() - dimsToReduce.size();
    if (numOfDimsLeft == 0) {
      return {1};
    }
    std::vector<int64_t> outputSize;
    outputSize.reserve(numOfDimsLeft);

    for (size_t i = 0; i < inputSize.size(); ++i) {
      if (std::find(dimsToReduce.begin(), dimsToReduce.end(), i) ==
          dimsToReduce.end()) {
        outputSize.push_back(inputSize[i]);
      }
    }
    return outputSize;
  }
}

std::vector<synapse_helpers::tensor> HandleReductionDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    std::vector<synTensor> inputs,
    const at::IntArrayRef dims,
    bool keepdim,
    const std::string& guid,
    std::vector<NodeAttr::NodeOutputAttr> output_attr,
    std::function<std::shared_ptr<
        void>(const int64_t, size_t&, int64_t, c10::optional<at::Scalar>)>
        fill_param_fn,
    c10::optional<at::Scalar> ord) {
  struct Reduction_Param {
    std::shared_ptr<void> param;
    size_t size{};
    std::vector<int64_t> shape;
  };
  auto dim = dims.vec();
  auto mask = std::bitset<64>();
  std::vector<int64_t> orig_shape{self.sizes().vec()};
  int ndims = orig_shape.size();
  auto num_outputs_required = output_attr.size();
  int num_tpc_outputs = 1;

  std::vector<synTensor> tensor_list;
  std::vector<synapse_helpers::tensor> flatten_input;
  std::vector<synapse_helpers::tensor> reshape_list;
  std::vector<synapse_helpers::tensor> tensor_itr;
  std::vector<Reduction_Param> param_list;

  HABANA_ASSERT(
      num_outputs_required <= 2, "Number of outputs is greater than 2.");

  ProcessDim(dim, ndims);
  auto use_flat_input = CheckDims(dim);
  if (use_flat_input) {
    // reshaped_self_sizes is used to hold appropriate input sizes
    std::vector<int64_t> reshaped_self_sizes;
    CombineDims(self, dim, keepdim, reshaped_self_sizes);

    auto flat_input = FlattenInput(op, graph, inputs[0], reshaped_self_sizes);
    flatten_input.emplace_back(std::move(flat_input));
    orig_shape = reshaped_self_sizes;
    ndims = reshaped_self_sizes.size();
  }

  for (const auto& i : dim) {
    mask.set(c10::maybe_wrap_dim(i, ndims, true));
  }

  int threshold = ndims == 0 ? -1 : 0;
  // reduce_Lp_fwd has unique fill_param which takes scalar as extra input
  for (int64_t dimIndex = orig_shape.size() - 1; dimIndex >= threshold;
       dimIndex--) {
    if (mask[dimIndex] || ndims == 0) {
      size_t size = 0;
      // Orig shape reduction can be ignored for ndims == 0 where dimIndex
      // becomes negative
      if (dimIndex >= 0) {
        orig_shape[dimIndex] = 1;
      }
      auto params = fill_param_fn(ndims, size, dimIndex, ord);
      param_list.push_back({params, size, orig_shape});
    }
  }

  // lambda for reduction op build node
  auto reduction_build_node =
      [op, &graph, &guid, param_list](
          std::vector<synTensor> input,
          const std::vector<NodeAttr::NodeOutputAttr>& attr,
          int param_index) -> std::vector<synapse_helpers::tensor> {
    return OpBackend::BuildNode(
        op,
        graph,
        {guid,
         std::move(input),
         attr,
         param_list[param_index].param.get(),
         param_list[param_index].size});
  };
  GuidOutCount(
      op,
      guid,
      param_list[0].shape,
      num_tpc_outputs,
      output_attr,
      num_outputs_required);

  auto reduce_output_attrs = [output_attr, param_list, num_tpc_outputs](
                                 const std::vector<int64_t>& outshape)
      -> std::vector<NodeAttr::NodeOutputAttr> {
    std::vector<NodeAttr::NodeOutputAttr> reduce_output_attrs{
        {outshape, output_attr[0].dtype}};
    // output shape for the intermediate node can be different from the shape in
    // output_attr passed to the handle function but the dtype will be same as
    // in output_attr
    for (int itr = 1; itr < num_tpc_outputs; itr++) {
      reduce_output_attrs.push_back(
          {param_list[0].shape, output_attr[itr].dtype});
    }
    return reduce_output_attrs;
  };
  size_t len = param_list.size();
  if (keepdim) {
    // When keepdim value is set to true
    if (len == 1) {
      auto op_out = reduction_build_node(
          use_flat_input ? std::vector<synTensor>{flatten_input[0].get()}
                         : std::move(inputs),
          output_attr,
          /*param_index*/ 0);

      return op_out;
    } else {
      auto op_out = reduction_build_node(
          std::move(inputs),
          reduce_output_attrs(param_list[0].shape),
          /*param_index*/ 0);

      tensor_list.emplace_back(op_out[0].get());
      // Iterating over the for loop when multiple dim values are passed
      // Output of previous iteration will be input of next iteration
      for (size_t i = 1; i <= len - 1; i++) {
        tensor_itr = reduction_build_node(
            {tensor_list[i - 1]},
            reduce_output_attrs(param_list[i].shape),
            /*param_index*/ i);
        tensor_list.emplace_back(tensor_itr[0].get());
      }
      for (unsigned int itr = 0; itr < num_outputs_required; itr++) {
        auto reshape = OpBackend::BuildReshape(
            op,
            graph,
            tensor_itr[itr].get(),
            output_attr[itr].sizes,
            output_attr[itr].dtype,
            output_attr[itr].final_result_index);
        // output of reshape is the output of this op
        reshape_list.emplace_back(std::move(reshape));
      }
      return reshape_list;
    }
  } else {
    // When keepdim value is set to false
    auto op_out = reduction_build_node(
        use_flat_input ? std::vector<synTensor>{flatten_input[0].get()}
                       : std::move(inputs),
        reduce_output_attrs(param_list[0].shape),
        /*param_index*/ 0);
    tensor_list.emplace_back(op_out[0].get());
    if (len == 1) {
      for (unsigned int itr = 0; itr < num_outputs_required; itr++) {
        auto reshape = OpBackend::BuildReshape(
            op,
            graph,
            op_out[itr].get(),
            output_attr[itr].sizes,
            output_attr[itr].dtype,
            output_attr[itr].final_result_index);
        // output of reshape is the output of this op
        reshape_list.emplace_back(std::move(reshape));
      }
      return reshape_list;
    } else {
      // Iterating over the for loop when multiple dim values are passed
      for (size_t i = 1; i <= len - 1; i++) {
        tensor_itr = reduction_build_node(
            {tensor_list[i - 1]},
            reduce_output_attrs(param_list[i].shape),
            /*param_index*/ i);
        tensor_list.emplace_back(tensor_itr[0].get());
      }
      // when reduction has to be done for all the dimension of input tensor,
      // TPC expects -[1,1,1,1] shape for 4d input but end outshape will be
      // {}-0d so reshape is used in this case as well
      for (unsigned int itr = 0; itr < num_outputs_required; itr++) {
        auto reshape = OpBackend::BuildReshape(
            op,
            graph,
            tensor_itr[itr].get(),
            output_attr[itr].sizes,
            output_attr[itr].dtype,
            output_attr[itr].final_result_index);
        reshape_list.emplace_back(std::move(reshape));
      }
      return reshape_list;
    }
  }
}

std::vector<synapse_helpers::tensor> HandleReductionMultiDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor syn_in,
    const std::string& guid,
    c10::IntArrayRef dimsToReduce,
    const int64_t inputRank,
    const bool keepdim,
    std::vector<NodeAttr::NodeOutputAttr> output_attr) {
  auto params = FillReductionParams(inputRank, dimsToReduce, keepdim);

  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision(guid, op->ScalarType()),
       {syn_in},
       std::move(output_attr),
       &params,
       sizeof(params)});
}
} // namespace habana
