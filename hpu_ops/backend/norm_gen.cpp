/**
* Copyright (c) 2021-2024 Intel Corporation
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
#include "generated/backend/linalg_vector_norm.h"
#include "generated/backend/norm.h"

#include "hpu_ops/backend/reduction_template.h"

constexpr const auto INF = std::numeric_limits<float>::infinity();

namespace habana {

namespace sh = synapse_helpers;

OutputMetaDataVector NormMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = {};
  meta.dtype = stack.size() == 3 ? stack.at(2).toScalarType()
                                 : stack_tensor(stack, 0).scalar_type();

  return {meta};
}

static sizes_vec NormOpOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim =
      stack.at(2).isNone() ? std::vector<int64_t>() : stack.at(2).toIntVector();

  const bool keepdim = stack.at(3).toBool();

  return ReductionOutputShape(self, dim, keepdim);
}

OutputMetaDataVector NormOpMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.dtype = (stack.size() >= 5 && !stack.at(4).isTensor())
      ? stack.at(4).toScalarType()
      : self.scalar_type();
  meta.shape = NormOpOutputShape(stack)[0];

  return {meta};
}

OutputMetaDataVector VecNormMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.dtype =
      stack.at(4).toOptional<at::ScalarType>().value_or(self.scalar_type());
  meta.shape = NormOpOutputShape(stack)[0];

  return {meta};
}

SharedMetaDataVector NormCommonSharedMeta(
    const int64_t& inputRank,
    const int64_t& outputRank,
    const at::ScalarType& dtype) {
  SharedMetaData normCommonSharedMeta{"reduce_Lp_multi_dim_fwd"};
  normCommonSharedMeta.inputs_data.emplace_back(inputRank, dtype);
  normCommonSharedMeta.outputs_data.emplace_back(outputRank, dtype);
  return {normCommonSharedMeta};
}

SharedMetaDataVector NormOpWithDtypeSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto self = stack_tensor(stack, 0);
  const bool keepdim = stack.at(3).toBool();
  const auto inputRank = self.dim();
  auto outputRank = inputRank;
  if (!keepdim) {
    if (stack.at(2).isNone()) {
      outputRank = 1;
    } else {
      const int64_t dims =
          static_cast<int64_t>(stack.at(2).toIntVector().size());
      outputRank = dims >= inputRank ? 1 : inputRank - dims;
    }
  }

  c10::ScalarType dtype;
  if (stack.size() >= 5 && !stack.at(4).isTensor() && !stack.at(4).isNone())
    dtype = stack.at(4).toScalarType();
  else
    dtype = self.scalar_type();

  return NormCommonSharedMeta(inputRank, outputRank, dtype);
}

SharedMetaDataVector NormOpScalarSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack_tensor(stack, 0);
  const auto inputRank = self.dim();
  const auto outputRank = 1;
  const auto dtype =
      stack.size() == 3 ? stack.at(2).toScalarType() : self.scalar_type();

  return NormCommonSharedMeta(inputRank, outputRank, dtype);
}

static ns_ReduceLpV2::ParamsV2 FillPFormNormOpParams(
    const int64_t ndims,
    at::IntArrayRef dims,
    bool keepDim,
    const at::Scalar& ord) {
  ns_ReduceLpV2::ParamsV2 params;
  (ns_Reduction::ParamsV2&)params = FillReductionParams(ndims, dims, keepDim);

  if (ord.isFloatingPoint()) {
    get<float>(params.p) = ord.to<float>();
    params.typeOfP = TYPE_P_IS_FLOAT;
  } else {
    get<int>(params.p) = ord.to<int>();
    params.typeOfP = TYPE_P_IS_INT;
  }

  return params;
}

static void VecNormCheck(
    const torch::Tensor& self,
    at::ScalarType dtype,
    const at::Scalar& ord,
    c10::IntArrayRef dim) {
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

static void NormCheck(at::ScalarType dtype) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dtype == torch::kBFloat16 || dtype == torch::kFloat,
      "norm: Expected input dtype to be Float or kBFloat16, but got ",
      dtype);
}

sh::tensor NormCommon(
    OpBackend* op,
    sh::graph& graph,
    synTensor input_tensor,
    at::ScalarType dtype,
    const torch::Tensor& self,
    at::IntArrayRef dim,
    const bool keepdim,
    const at::Scalar& ord,
    const std::vector<NodeAttr::NodeOutputAttr>& output_attr,
    const bool is_vec_norm) {
  auto p = (ord.isFloatingPoint()) ? ord.toFloat() : ord.toInt();
  auto self_shape = self.sizes().vec();

  if (is_vec_norm)
    VecNormCheck(self, dtype, ord, dim);
  else
    NormCheck(dtype);

  if (self.numel() == 0) {
    at::Scalar s = (p < 0) && is_vec_norm ? INF : 0;
    return OpBackend::BuildConstant(op, graph, s, dtype, 1, 0);
  }

  auto params = FillPFormNormOpParams(self.dim(), dim, keepdim, ord);
  auto reduce_lp_output = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("reduce_Lp_multi_dim_fwd", dtype),
       {input_tensor},
       output_attr,
       &params,
       sizeof(params)});

  return std::move(reduce_lp_output.at(0));
}

void VecNormOp::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  auto dim =
      stack.at(2).isNone() ? c10::DimVector{} : stack.at(2).toDimVector();
  const at::Scalar ord = optional_ord.value_or(2);
  const bool keepdim = stack.at(3).toBool();

  auto meta = VecNormMeta(stack)[0];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      meta.dtype == torch::kBFloat16 || meta.dtype == torch::kFloat,
      "linalg.vector_norm: Expected input dtype to be Float or kBFloat16, but got ",
      meta.dtype);

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      meta.dtype,
      self,
      dim,
      keepdim,
      ord,
      {{meta.shape, meta.dtype, 0}},
      true /* vec_norm */);
  syn_out(0) = std::move(result);
}

void NormOpWithDtype::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  auto dim = stack.at(2).toDimVector();
  const at::Scalar ord = optional_ord.value_or(2);
  const bool keepdim = stack.at(3).toBool();
  auto meta = NormOpMeta(stack)[0];

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      meta.dtype,
      self,
      dim,
      keepdim,
      ord,
      {{meta.shape, meta.dtype, 0}},
      false /* norm */);
  syn_out(0) = std::move(result);
}

void NormOpScalar::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto ord = stack.at(1).toScalar();
  auto meta = NormMeta(stack)[0];

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      meta.dtype,
      self,
      {},
      false,
      ord,
      {{meta.shape, meta.dtype, 0}},
      false /* norm */);
  syn_out(0) = std::move(result);
}

void NormOpScalarWithDtype::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optional_ord = stack.at(1).toOptional<at::Scalar>();
  const at::Scalar ord = optional_ord.value_or(2);
  auto meta = NormMeta(stack)[0];

  auto result = NormCommon(
      this,
      graph,
      syn_in(0),
      meta.dtype,
      self,
      {},
      false,
      ord,
      {{meta.shape, meta.dtype, 0}},
      false /* norm */);
  syn_out(0) = std::move(result);
}

} // namespace habana
