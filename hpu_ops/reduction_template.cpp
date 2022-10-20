/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "reduction_template.h"
#include "habana_kernels/lowering_util.h"
#include "hpu_op_helper.h"

namespace habana {

static std::vector<int64_t> get_dims(
    at::Stack stack,
    at::optional<uint8_t> dim_index) {
  std::vector<int64_t> dims;
  auto dim_ival =
      dim_index.has_value() ? stack.at(dim_index.value()) : at::IValue();
  if (dim_ival.isInt()) {
    dims = {dim_ival.toInt()};
  } else if (dim_ival.isIntList()) {
    dims = dim_ival.toIntVector();
  } else {
    HABANA_ASSERT(
        dim_ival.isNone(),
        "Reduction op dims can be int, int list or none but got ",
        dim_ival.tagKind());
  }
  return dims;
}

static bool get_keepdim(at::Stack stack, at::optional<uint8_t> keepdim_index) {
  return keepdim_index.has_value() ? stack.at(keepdim_index.value()).toBool()
                                   : false;
}

static at::optional<at::ScalarType> get_dtype(
    at::Stack stack,
    at::optional<uint8_t> dtype_index) {
  return dtype_index.has_value()
      ? stack.at(dtype_index.value()).toOptional<at::ScalarType>()
      : at::nullopt;
}

static at::IntArrayRef optional_to_arrayref(const c10::optional<int64_t>& opt) {
  return opt.has_value() ? opt.value() : at::IntArrayRef{};
}

static at::IntArrayRef optional_to_arrayref(
    const c10::OptionalIntArrayRef& opt) {
  return opt.has_value() ? opt.value() : at::IntArrayRef{};
}

sizes_vec ReductionOutputShape(
    const at::Tensor& self,
    at::optional<int64_t> dims,
    bool keepdim) {
  return ReductionOutputShape(self, optional_to_arrayref(dims), keepdim);
}

sizes_vec ReductionOutputShape(
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim) {
  at::DimVector shape =
      at::meta::get_reduction_shape(self, optional_to_arrayref(dims), keepdim);
  return {std::vector<int64_t>(shape.begin(), shape.end())};
}

template <>
at::Tensor ReductionFrontendTemplate<at::Tensor>::get_result_overrideable() {
  const auto& stack = LazyOp<at::Tensor>::get_inputs();
  const torch::Tensor& self = stack_tensor(stack, 0);

  HABANA_ASSERT(!is_outfn_, "Unexpected output op variant");
  auto dtype_helper =
      habana_helpers::DTypeHelper::unary_op_with_optional_int_to_long_promotion(
          stack, c10::nullopt, get_dtype(stack, m_dtype_index), true);

  return at::native::create_reduction_result(
      self,
      get_dims(stack, m_dim_index),
      get_keepdim(stack, m_keepdim_index),
      dtype_helper.get_result_dtype());
}

template <>
at::Tensor& ReductionFrontendTemplate<at::Tensor&>::get_result_overrideable() {
  throw std::invalid_argument("Tensor ref should not be created.");
}

template <>
void ReductionFrontendTemplate<at::Tensor>::Validate() {
  const auto& stack = LazyOp<at::Tensor>::get_inputs();
  c10::optional<const at::IValue*> output = is_outfn_
      ? c10::make_optional<const at::IValue*>(&stack.back())
      : c10::nullopt;
  auto dtype_helper =
      habana_helpers::DTypeHelper::unary_op_with_optional_int_to_long_promotion(
          stack, output, get_dtype(stack, m_dtype_index), true);
}

template <>
void ReductionFrontendTemplate<at::Tensor&>::Validate() {
  const auto& stack = LazyOp<at::Tensor&>::get_inputs();
  c10::optional<const at::IValue*> output =
      c10::make_optional<const at::IValue*>(
          is_outfn_ ? &stack.back() : &stack.front());
  auto dtype_helper =
      habana_helpers::DTypeHelper::unary_op_with_optional_int_to_long_promotion(
          stack, output, get_dtype(stack, m_dtype_index), true);
}

ReductionBackendTemplate::ReductionBackendTemplate(
    int device_id,
    const std::string& guid,
    at::ScalarType scalar_type,
    std::vector<int> res_ids,
    std::vector<int> inplace_ids,
    std::vector<int> scalar_ids,
    bool is_outfn)
    : OpBackend(
          device_id,
          guid,
          scalar_type,
          std::move(res_ids),
          std::move(inplace_ids),
          std::move(scalar_ids),
          is_outfn) {}

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

  auto shape = ComputeOutputShapes(stack).empty()
      ? ReductionOutputShape(self, dims, keepdim)[0]
      : ComputeOutputShapes(stack)[0];
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{{shape, ScalarType(), 0}};

  auto result = HandleReductionDimAndKeepdim(
      this, graph, self, {input}, dims, keepdim, GetGuid(), output_attrs);

  syn_out(0) = std::move(result[0]);
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

  if (!dtype.has_value() and at::isIntegralType(self.scalar_type(), true)) {
    dtype_val = at::kInt;
  }

  if (dtype_val == self.scalar_type()) {
    return c10::nullopt;
  }

  op->SetScalarType(dtype_val);

  // Update guid with the dtype to be used
  std::string guid = op->GetGuid();
  op->SetGuid(update_guid_dtype(guid, dtype_val));

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
      1,
      std::multiplies<int>());
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
    size_t num_outputs) {
  // TPC guids which returns two outputs
  std::vector<std::string> multi_output_reduce_ops = {
      "reduce_min_fwd",
      "reduce_max_fwd",
      "reduce_Lp_fwd",
      "reduce_log_sum_exp_fwd",
      "reduce_log_sum_fwd"};
  for (size_t i = 0; i < multi_output_reduce_ops.size(); i++) {
    if (guid.find(multi_output_reduce_ops[i]) != std::string::npos) {
      num_tpc_outputs = 2;
      // when caller needs only one output but TPC retuns two output
      if (num_outputs == 1)
        output_attr.push_back({retain_ten_shape, op->ScalarType()});
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
  auto num_outputs = output_attr.size();
  int num_tpc_outputs = 1;

  std::vector<synTensor> tensor_list;
  std::vector<synapse_helpers::tensor> flatten_input;
  std::vector<synapse_helpers::tensor> reshape_list;
  std::vector<synapse_helpers::tensor> tensor_itr;
  std::vector<Reduction_Param> param_list;

  HABANA_ASSERT(num_outputs <= 2, "Number of outputs is greater than 2.");

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

  // reduce_Lp_fwd has unique fill_param which takes scalar as extra input
  for (int64_t dimIndex = orig_shape.size() - 1; dimIndex >= 0; dimIndex--) {
    if (mask[dimIndex]) {
      orig_shape[dimIndex] = 1;
      size_t size = 0;
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
      op, guid, param_list[0].shape, num_tpc_outputs, output_attr, num_outputs);

  auto reduce_output_attrs =
      [output_attr, param_list, num_tpc_outputs](std::vector<int64_t> outshape)
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
      for (unsigned int itr = 0; itr < num_outputs; itr++) {
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
      for (unsigned int itr = 0; itr < num_outputs; itr++) {
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
      for (unsigned int itr = 0; itr < num_outputs; itr++) {
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
} // namespace habana
