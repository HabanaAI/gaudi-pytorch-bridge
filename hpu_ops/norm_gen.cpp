/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/linalg_vector_norm.h"
#include "generated/norm.h"
#include "hpu_op_helper.h"
#include "reduction_template.h"

#define INF std::numeric_limits<float>::infinity()
namespace habana {

sizes_vec NormOutputShape(const at::Stack&) {
  return {{}};
}

// Second param is unused so neglecting it
sizes_vec NormOpOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim =
      stack.at(2).isNone() ? std::vector<int64_t>() : stack.at(2).toIntVector();

  const bool keepdim = stack.at(3).toBool();

  return ReductionOutputShape(self, dim, keepdim);
}

std::shared_ptr<void> FillPFormNormOpParams(
    const int64_t ndim,
    size_t& size,
    int64_t index,
    c10::optional<at::Scalar> opt_ord) {
  const at::Scalar ord = opt_ord.value_or(2);
  PARAMS_STUB(ns_ReduceLpV2::Params);
  auto reduction_dim = ndim - 1 - index;
  params->reductionDimension = reduction_dim;
  if (ord.isFloatingPoint()) {
    get<float>(params->p) = ord.to<float>();
    params->typeOfP = TYPE_P_IS_FLOAT;
  } else {
    get<int>(params->p) = ord.to<int>();
    params->typeOfP = TYPE_P_IS_INT;
  }
  return params;
}

template <>
LazyNormOp<at::Tensor>::LazyNormOp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyNormOp<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& self = inputs.at(0).toTensor();
  auto shape = get_out_shapes()[0];
  const at::ScalarType& dtype =
      inputs.at(4).isNone() ? self.scalar_type() : inputs.at(4).toScalarType();
  return habana_lazy::empty_hpu_lazy(
      shape, self.options().dtype(dtype), self.suggest_memory_format(), false);
}

void NormHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto p = stack.at(1).toScalar();
  const auto dtype = stack.at(2).toScalarType();
  auto outshape = self.sizes();
  auto n_dims = self.dim();

  auto input = syn_in(0);
  auto input_in_dtype = HandleReductionDtype(this, graph, self, input, dtype);
  if (input_in_dtype.has_value()) {
    input = input_in_dtype.value().get();
  }

  if (p.toFloat() == 2.0) {
    if (n_dims <= 1 || self.sizes()[0] == 1) {
      auto mul = BuildOp(
          graph,
          MULT_GUID + habana_helpers::name_suffix_from_type(dtype),
          {input, input},
          {{outshape, dtype}});

      std::vector<synTensor> reduction_inputs = {mul[0].get()};
      std::vector<synapse_helpers::tensor> reshape;

      if (n_dims > 1) {
        auto reshape_outshape = self.numel();
        reshape.emplace_back(
            ReshapeHelper(graph, reduction_inputs[0], reshape_outshape, dtype));
        reduction_inputs = {reshape[0].get()};
      }

      ns_Reduction::Params reduce_params{};
      reduce_params.reductionDimension = 0;
      auto sum = BuildOp(
          graph,
          "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(dtype),
          reduction_inputs,
          {{1, dtype}},
          &reduce_params,
          sizeof(reduce_params));

      auto sqrt = BuildOp(
          graph,
          "sqrt_fwd_" + habana_helpers::name_suffix_from_type(dtype),
          {sum[0].get()},
          {{1, dtype, 0}});

      syn_out(0) = std::move(sqrt[0]);

    } else {
      auto norm = BuildOp(
          graph,
          "frobenius_norm_fwd_" + habana_helpers::name_suffix_from_type(dtype),
          {input},
          {{1, dtype, 0}});

      syn_out(0) = std::move(norm[0]);
    }

  } else {
    auto reshape_outshape = self.numel();
    std::vector<synapse_helpers::tensor> reshape;
    if (n_dims > 1) {
      reshape.emplace_back(
          ReshapeHelper(graph, input, reshape_outshape, dtype));
      input = reshape[0].get();
    }

    ns_LpNormKernel::Params lpnorm_params{};
    lpnorm_params.p = p.toFloat();
    lpnorm_params.dim = 0;
    lpnorm_params.eps = 0;
    auto norm = BuildOp(
        graph,
        "lpnorm_fwd_" + habana_helpers::name_suffix_from_type(dtype),
        {input},
        {{reshape_outshape, dtype}, {reshape_outshape, dtype}},
        &lpnorm_params,
        sizeof(lpnorm_params));

    auto reciprocal = BuildOp(
        graph,
        "reciprocal_fwd_" + habana_helpers::name_suffix_from_type(dtype),
        {norm[1].get()},
        {{reshape_outshape, dtype}});

    synSliceParamsNDims slice_params{};
    slice_params.axes[0] = 0;
    slice_params.starts[0] = 0;
    slice_params.ends[0] = 1;
    slice_params.steps[0] = 1;
    auto slice = BuildOp(
        graph,
        "slice_" + habana_helpers::name_suffix_from_type(dtype),
        {reciprocal[0].get()},
        {{1, dtype, 0}},
        &slice_params,
        sizeof(slice_params));

    syn_out(0) = std::move(slice[0]);
  }
}

static synapse_helpers::tensor L0NormPreprocess(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef inputshape,
    const at::ScalarType& dtype) {
  auto zero = OpBackend::BuildConstant(op, graph, 0, dtype, inputshape);
  auto compare = OpBackend::BuildNode(
      op,
      graph,
      {"equal_fwd_" + habana_helpers::name_suffix_from_type(dtype),
       {input[0], zero.get()},
       {{inputshape, torch::kBool}}});

  auto not_equal = OpBackend::BuildNode(
      op,
      graph,
      {"not_fwd_i8", {compare[0].get()}, {{inputshape, torch::kBool}}});

  return OpBackend::BuildCast(
      op, graph, not_equal[0].get(), inputshape, torch::kBool, dtype);
}

static synapse_helpers::tensor NegPosInfNormPreprocess(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef inputshape,
    const at::ScalarType& dtype) {
  auto abs = OpBackend::BuildNode(
      op,
      graph,
      {"abs_fwd_" + habana_helpers::name_suffix_from_type(dtype),
       std::move(input),
       {{inputshape, dtype}}});

  return std::move(abs.at(0));
}

void VecNormCheck(
    const torch::Tensor& self,
    const at::ScalarType& dtype,
    at::Scalar ord,
    const std::vector<int64_t>& dim) {
  auto p = (ord.isFloatingPoint()) ? ord.toFloat() : ord.toInt();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dtype == torch::kBFloat16 || dtype == torch::kFloat,
      "linalg.vector_norm: Expected input dtype to be Float or kBFloat16, but got ",
      dtype);

  if (self.numel() == 0) {
    TORCH_CHECK(
        p >= 0,
        "linalg.vector_norm of negative order cannot be performed on an empty tensor");
    if (p == INF) {
      bool has_identity = true;
      if (dim.size() == 0) {
        has_identity = false;
      } else {
        for (unsigned i = 0; i < dim.size(); ++i) {
          if (self.size(dim[i]) == 0) {
            has_identity = false;
            break;
          }
        }
      }
      TORCH_CHECK(
          has_identity,
          "linalg.vector_norm cannot compute the infinity norm on an empty ",
          "dimension because the operation does not have an identity");
    }
  }
}

void NormCheck(const at::ScalarType& dtype) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dtype == torch::kBFloat16 || dtype == torch::kFloat,
      "norm: Expected input dtype to be Float or kBFloat16, but got ",
      dtype);
}

static synapse_helpers::tensor NormCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input_tensor,
    const at::ScalarType& dtype,
    const torch::Tensor& self,
    std::vector<int64_t> dim,
    const bool keepdim,
    at::Scalar ord,
    std::vector<NodeAttr::NodeOutputAttr> output_attr,
    const bool is_vec_norm) {
  auto p = (ord.isFloatingPoint()) ? ord.toFloat() : ord.toInt();
  auto norm_ord = ord.toFloat();
  auto self_shape = self.sizes().vec();
  struct vec_norm_inputs {
    std::string guid;
    std::function<synapse_helpers::tensor(
        OpBackend*,
        synapse_helpers::graph&,
        std::vector<synTensor>,
        const at::IntArrayRef,
        const at::ScalarType&)>
        pre_fn{};
  };
  if (is_vec_norm)
    VecNormCheck(self, dtype, ord, dim);
  else
    NormCheck(dtype);

  if (self.numel() == 0) {
    at::Scalar s = (p < 0) && is_vec_norm ? INF : 0;
    return OpBackend::BuildConstant(op, graph, s, dtype, 1, 0);
  }

  std::map<float, vec_norm_inputs> mod_inputs = {
      {0.0, {"reduce_sum_fwd_", L0NormPreprocess}},
      {1.0, {"reduce_L1_fwd_"}},
      {2.0, {"reduce_L2_fwd_"}},
      {INF, {"reduce_max_fwd_", NegPosInfNormPreprocess}},
      {-INF, {"reduce_min_fwd_", NegPosInfNormPreprocess}}};

  // Using Lp_fwd if the ord value is not present in the map
  if (mod_inputs.find(norm_ord) == mod_inputs.end()) {
    auto norm_itr = HandleReductionDimAndKeepdim(
        op,
        graph,
        self,
        {input_tensor},
        dim,
        keepdim,
        "reduce_Lp_fwd_" + habana_helpers::name_suffix_from_type(dtype),
        output_attr,
        FillPFormNormOpParams,
        ord);
    return std::move(norm_itr.at(0));
  } else {
    auto inputs = mod_inputs[norm_ord];
    auto norm_itr = HandleReductionDimAndKeepdim(
        op,
        graph,
        self,
        {inputs.pre_fn
             ? inputs.pre_fn(op, graph, {input_tensor}, self_shape, dtype).get()
             : input_tensor},
        dim,
        keepdim,
        inputs.guid + habana_helpers::name_suffix_from_type(dtype),
        output_attr);
    return std::move(norm_itr.at(0));
  }
}

void VecNormOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_dtype = stack.at(4).toOptional<at::ScalarType>();
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  std::vector<int64_t> dim =
      stack.at(2).isNone() ? std::vector<int64_t>() : stack.at(2).toIntVector();
  const at::Scalar ord = optional_ord.value_or(2);
  const at::ScalarType& dtype = optional_dtype.value_or(ScalarType());
  const bool keepdim = stack.at(3).toBool();

  auto output_shape = NormOpOutputShape(stack)[0];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dtype == torch::kBFloat16 || dtype == torch::kFloat,
      "linalg.vector_norm: Expected input dtype to be Float or kBFloat16, but got ",
      dtype);

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      dtype,
      self,
      dim,
      keepdim,
      ord,
      {{output_shape, dtype, 0}},
      true /* vec_norm */);
  syn_out(0) = std::move(result);
}

void NormOpWithDtype::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  std::vector<int64_t> dim = stack.at(2).toIntVector();
  const at::Scalar ord = optional_ord.value_or(2);
  const at::ScalarType& dtype = stack.at(4).toScalarType();
  const bool keepdim = stack.at(3).toBool();
  auto output_shape = NormOpOutputShape(stack)[0];

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      dtype,
      self,
      dim,
      keepdim,
      ord,
      {{output_shape, dtype, 0}},
      false /* norm */);
  syn_out(0) = std::move(result);
}

void NormOpWithOutDtype::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  std::vector<int64_t> dim = stack.at(2).toIntList().vec();
  const at::Scalar ord = optional_ord.value_or(2);
  const bool keepdim = stack.at(3).toBool();
  auto output_shape = NormOpOutputShape(stack)[0];

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      ScalarType(),
      self,
      dim,
      keepdim,
      ord,
      {{output_shape, ScalarType(), 0}},
      false /* norm */);
  syn_out(0) = std::move(result);
}

void NormOpScalar::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto ord = stack.at(1).toScalar();

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      ScalarType(),
      self,
      {},
      false,
      ord,
      {{1, ScalarType(), 0}},
      false /* norm */);
  syn_out(0) = std::move(result);
}
} // namespace habana
