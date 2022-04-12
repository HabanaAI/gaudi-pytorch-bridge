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

inline at::ScalarType get_dtype_from_self(
    const at::Tensor& self,
    const at::optional<at::ScalarType>& dtype,
    bool promote_integers) {
  if (dtype.has_value()) {
    return dtype.value();
  }
  at::ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return at::kLong;
  }
  return src_type;
}

static at::IntArrayRef optional_to_arrayref(const c10::optional<int64_t>& opt) {
  return opt.has_value() ? opt.value() : at::IntArrayRef{};
}

static at::IntArrayRef optional_to_arrayref(
    const c10::optional<at::IntArrayRef>& opt) {
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
    at::optional<at::IntArrayRef> dims,
    bool keepdim) {
  at::DimVector shape =
      at::meta::get_reduction_shape(self, optional_to_arrayref(dims), keepdim);
  return {std::vector<int64_t>(shape.begin(), shape.end())};
}

template <>
at::Tensor ReductionFrontendTemplate<at::Tensor>::get_result_overrideable() {
  const auto& stack = LazyOp<at::Tensor>::get_inputs();
  const torch::Tensor& self = stack_tensor(stack, 0);

  return at::native::create_reduction_result(
      self,
      get_dims(stack, m_dim_index),
      get_keepdim(stack, m_keepdim_index),
      get_dtype_from_self(self, get_dtype(stack, m_dtype_index), true));
}

template <>
at::Tensor& ReductionFrontendTemplate<at::Tensor&>::get_result_overrideable() {
  throw std::invalid_argument("Tensor ref should not be created.");
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
  synapse_helpers::tensor& input = GetSynInputs()[0];

  // Extract dtype and cast self to the supplied dtype
  auto dtype = get_dtype(stack, m_dtype_index);
  input = HandleReductionDtype(this, graph, self, std::move(input), dtype);

  // Extract dims
  auto dims = get_dims(stack, m_dim_index);

  // Extract keepdim
  bool keepdim = get_keepdim(stack, m_keepdim_index);

  auto shape = ComputeOutputShapes(stack, true).empty()
      ? ReductionOutputShape(self, dims, keepdim)[0]
      : ComputeOutputShapes(stack, true)[0];
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{{shape, ScalarType(), 0}};

  auto result = HandleReductionDimAndKeepdim(
      this, graph, self, {input.get()}, dims, keepdim, GetGuid(), output_attrs);

  syn_out(0) = std::move(result[0]);
}

synapse_helpers::tensor HandleReductionDtype(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synapse_helpers::tensor syn_in,
    at::optional<at::ScalarType> dtype) {
  auto dtype_val = dtype.value_or(self.scalar_type());

  if (!dtype.has_value() and at::isIntegralType(self.scalar_type(), true)) {
    dtype_val = at::kInt;
  }

  if (dtype_val == self.scalar_type()) {
    return syn_in;
  }

  op->SetScalarType(dtype_val);

  // Update guid with the dtype to be used
  std::string guid = op->GetGuid();
  op->SetGuid(update_guid_dtype(guid, dtype_val));

  return OpBackend::BuildCast(
      op, graph, syn_in.get(), self.sizes(), self.scalar_type(), dtype_val);
}

static std::shared_ptr<void> ReductionOpParams(
    const int ndim,
    size_t& size,
    int64_t index) {
  PARAMS_STUB(ns_Reduction::Params);
  auto reduction_dim = ndim - 1 - index;
  params->reductionDimension = reduction_dim;
  return params;
}

std::vector<synapse_helpers::tensor> HandleReductionDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    std::vector<synTensor> inputs,
    const at::IntArrayRef dims,
    bool keepdim,
    const std::string& guid,
    const std::vector<NodeAttr::NodeOutputAttr>& output_attr) {
  struct Parameters {
    std::shared_ptr<void> param_list;
    size_t size_list{};
    std::vector<int64_t> shape_list;
  };
  auto dim = dims.vec();
  auto mask = std::bitset<64>();
  std::vector<int64_t> orig_shape{self.sizes().vec()};
  const int ndims = orig_shape.size();
  auto num_outputs = output_attr.size();
  std::vector<synTensor> tensor_list;
  std::vector<synapse_helpers::tensor> reshape_list;
  std::vector<synapse_helpers::tensor> tensor_itr;
  std::vector<Parameters> parameters;
  // When dim=[], reduce all dimensions based on keepdim value
  if (0 == dim.size()) {
    for (int i = 0; i < ndims; ++i) {
      dim.push_back(i);
    }
  }
  HABANA_ASSERT(num_outputs <= 2, "Number of outputs is greater than 2.");
  auto reduc_output_attrs =
      [](sizes_vec outshapes,
         std::vector<at::ScalarType> dtypes,
         int num_out) -> std::vector<NodeAttr::NodeOutputAttr> {
    std::vector<NodeAttr::NodeOutputAttr> reduc_output_attrs;
    for (int itr = 0; itr < num_out; itr++) {
      reduc_output_attrs.push_back({outshapes.at(itr), dtypes.at(itr)});
    }
    return reduc_output_attrs;
  };
  for (const auto& i : dim) {
    mask.set(c10::maybe_wrap_dim(i, ndims, true));
  }
  for (int64_t dimIndex = orig_shape.size() - 1; dimIndex >= 0; dimIndex--) {
    if (mask[dimIndex]) {
      orig_shape[dimIndex] = 1;
      size_t size = 0;
      auto params = ReductionOpParams(ndims, size, dimIndex);
      parameters.push_back({params, size, orig_shape});
    }
  }
  auto retain_ten_shape =
      num_outputs > 1 ? parameters[0].shape_list : self.sizes().vec();
  size_t len = parameters.size();
  // NOTE: 1. Need to handle when TPC returns two outputs and pytorch
  // returns one output.
  // 2. Flatten the input when dim is none for certain ops.
  if (keepdim) {
    // When keepdim value is set to true
    if (len == 1) {
      auto op_out = OpBackend::BuildNode(
          op,
          graph,
          {guid,
           std::move(inputs),
           output_attr,
           parameters[0].param_list.get(),
           parameters[0].size_list});
      return op_out;
    } else {
      auto op_out = OpBackend::BuildNode(
          op,
          graph,
          {guid,
           std::move(inputs),
           reduc_output_attrs(
               {parameters[0].shape_list, retain_ten_shape},
               {output_attr[0].dtype, output_attr[1].dtype},
               num_outputs),
           parameters[0].param_list.get(),
           parameters[0].size_list});
      tensor_list.emplace_back(op_out[0].get());
      // Iterating over the for loop when multiple dim values are passed
      for (size_t i = 1; i <= len - 1; i++) {
        tensor_itr = OpBackend::BuildNode(
            op,
            graph,
            {guid,
             {tensor_list[i - 1]},
             reduc_output_attrs(
                 {parameters[i].shape_list, retain_ten_shape},
                 {output_attr[0].dtype, output_attr[1].dtype},
                 num_outputs),
             parameters[i].param_list.get(),
             parameters[i].size_list});
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
    auto op_out = OpBackend::BuildNode(
        op,
        graph,
        {guid,
         std::move(inputs),
         reduc_output_attrs(
             {parameters[0].shape_list, retain_ten_shape},
             {output_attr[0].dtype, output_attr[1].dtype},
             num_outputs),
         parameters[0].param_list.get(),
         parameters[0].size_list});
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
        tensor_itr = OpBackend::BuildNode(
            op,
            graph,
            {guid,
             {tensor_list[i - 1]},
             reduc_output_attrs(
                 {parameters[i].shape_list, retain_ten_shape},
                 {output_attr[0].dtype, output_attr[1].dtype},
                 num_outputs),
             parameters[i].param_list.get(),
             parameters[i].size_list});
        tensor_list.emplace_back(tensor_itr[0].get());
      }
      // output of reshape is the output of this op
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
  }
}
} // namespace habana
