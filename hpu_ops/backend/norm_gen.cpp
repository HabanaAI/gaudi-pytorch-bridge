/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
