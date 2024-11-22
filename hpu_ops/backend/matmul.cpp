/**
 * Copyright (c) 2024 Intel Corporation
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

#include "hpu_ops/matmul.h"
#include "backend/helpers/runtime_config.h"
#include "hpu_ops/common/batched_matmul_output_shape.h"

namespace sh = synapse_helpers;

namespace habana {

OutputMetaDataVector MatmulMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);
  const auto& other = stack_tensor(stack, 1);

  OutputMetaDataVector meta(1);
  meta.at(0).shape =
      getBatchMatmulOutShape(self.sizes(), other.sizes(), false, false);
  meta.at(0).dtype = self.scalar_type();
  return meta;
}

Matmul::Matmul(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "batch_gemm", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(MatmulMeta);
}

void Matmul::AddNode(sh::graph& graph, const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);
  const auto self_dim = self.dim();
  const auto self_dtype = self.scalar_type();
  const auto& other = stack_tensor(stack, 1);
  const auto other_dim = other.dim();

  synGEMMParams params{false, false};
  const auto meta = MatmulMeta(stack)[0];

  // TODO: https://jira.habana-labs.com/browse/SW-208805
  // Below logic is copied from lazy mode, where it performs well.
  // It's worth to simplify it in future, e.g. by handling some
  // optimizations as GC passes.

  if (self_dim == other_dim or (self_dim == 2 and other_dim == 1)) {
    if (self_dim == other_dim) {
      if (self_dim == 1) {
        SetGuid(get_guid_with_precision("dot_fwd", self_dtype));
      } else if (self_dim == 2) {
        SetGuid("gemm");
      }
    } else {
      SetGuid(get_guid_with_precision("mv_fwd", self_dtype));
    }
    auto output = OpBackend::BuildNode(
        this,
        graph,
        {guid_, {syn_in(0), syn_in(1)}, {{meta.shape, meta.dtype, 0}}});

    syn_out(0) = std::move(output[0]);
    return;
  }

  bool reshape_3d_2d = habana_helpers::IsMatmul3d2dReshapeEnabled();
  auto gemm_output_shape = meta.shape;

  if (self_dim == 1 and other_dim == 2) {
    auto expanded_sizes = self.sizes().vec();
    expanded_sizes.insert(expanded_sizes.cbegin(), 1);
    gemm_output_shape.insert(gemm_output_shape.cbegin(), 1);
    auto reshaped_self =
        ReshapeHelper(graph, syn_in(0), expanded_sizes, self_dtype);

    auto output = OpBackend::BuildNode(
        this,
        graph,
        {"gemm",
         {reshaped_self.get(), syn_in(1)},
         {{gemm_output_shape, meta.dtype}}});

    syn_out(0) = std::move(
        ReshapeHelper(graph, output[0].get(), meta.shape, self_dtype, 0));
  } else if (self_dim >= 3 and other_dim == 1) {
    auto expanded_sizes = other.sizes().vec();
    expanded_sizes.push_back(1);
    gemm_output_shape.push_back(1);
    auto reshaped_other =
        ReshapeHelper(graph, syn_in(1), expanded_sizes, self_dtype);

    auto output = OpBackend::BuildNode(
        this,
        graph,
        {"batch_gemm",
         {syn_in(0), reshaped_other.get()},
         {{gemm_output_shape, meta.dtype}}});

    syn_out(0) = std::move(
        ReshapeHelper(graph, output[0].get(), meta.shape, self_dtype, 0));
  } else if ((self_dim == 1 || self_dim == 2) && other_dim >= 3) {
    std::vector<sh::tensor> expanded_tensor;
    auto expanded_sizes = self.sizes().vec();
    if (self_dim == 1) {
      expanded_sizes.push_back(1);
      gemm_output_shape.push_back(1);

      expanded_tensor.emplace_back(
          ReshapeHelper(graph, syn_in(0), expanded_sizes, self_dtype));
    } else {
      synTransposeParamsNDims params;
      params.tensorDim = 2;
      for (int i = 0; i < HABANA_DIM_MAX; i++) {
        params.permutation[i] = static_cast<TransposePermutationDim>(i);
      }
      std::swap(params.permutation[0], params.permutation[1]);
      std::swap(expanded_sizes[0], expanded_sizes[1]);
      expanded_tensor.emplace_back(std::move(BuildNode(
          this,
          graph,
          {"transpose",
           {syn_in(0)},
           {{expanded_sizes, self_dtype}},
           &params,
           sizeof(params)})[0]));
    }

    synGEMMParams gemm_params{true, false};

    auto output = OpBackend::BuildNode(
        this,
        graph,
        {"batch_gemm",
         {syn_in(1), expanded_tensor.back().get()},
         {{gemm_output_shape, meta.dtype}},
         &gemm_params,
         sizeof(gemm_params)});

    if (self_dim == 1) {
      syn_out(0) = std::move(
          ReshapeHelper(graph, output[0].get(), meta.shape, self_dtype, 0));
    } else {
      synTransposeParamsNDims params;
      params.tensorDim = gemm_output_shape.size();
      for (int i = 0; i < HABANA_DIM_MAX; i++) {
        params.permutation[i] = static_cast<TransposePermutationDim>(i);
      }
      std::swap(params.permutation[0], params.permutation[1]);
      std::swap(expanded_sizes[0], expanded_sizes[1]);
      syn_out(0) = std::move(BuildNode(
          this,
          graph,
          {"transpose",
           {output[0].get()},
           {{meta.shape, self_dtype, 0}},
           &params,
           sizeof(params)})[0]);
    }
  } else if (
      (self_dim == 4 && other_dim == 3) || (self_dim == 3 && other_dim == 4)) {
    int64_t n = self.size(-2);
    int64_t m1 = self.size(-1);
    at::IntArrayRef batch_tensor1(self.sizes().data(), self_dim - 2);
    int64_t m2 = other.size(-2);
    int64_t p = other.size(-1);
    at::IntArrayRef batch_tensor2(other.sizes().data(), other_dim - 2);

    std::vector<int64_t> expand_batch_portion =
        at::infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> self_expand_size(expand_batch_portion);
    self_expand_size.insert(self_expand_size.end(), {n, m1});

    std::vector<int64_t> other_expand_size(expand_batch_portion);
    other_expand_size.insert(other_expand_size.end(), {m2, p});

    auto broadcast_self =
        BroadcastHelper(graph, syn_in(0), self_expand_size, self_dtype);

    auto broadcast_other =
        BroadcastHelper(graph, syn_in(1), other_expand_size, self_dtype);

    syn_out(0) = std::move(OpBackend::BuildNode(
        this,
        graph,
        {"batch_gemm",
         {broadcast_self.get(), broadcast_other.get()},
         {{gemm_output_shape, meta.dtype, 0}}})[0]);
  } else if (reshape_3d_2d && (self_dim == 3) && (other_dim == 2)) {
    auto self_sizes = self.sizes().vec();
    auto other_sizes = other.sizes().vec();
    std::vector<int64_t> shape_in{self_sizes[0] * self_sizes[1], self_sizes[2]};

    auto reshaped_self = ReshapeHelper(graph, syn_in(0), shape_in, self_dtype);
    std::vector<int64_t> gemm_output_shape{
        self_sizes[0] * self_sizes[1], other_sizes[2]};

    auto output = OpBackend::BuildNode(
        this,
        graph,
        {"gemm",
         {reshaped_self.get(), syn_in(1)},
         {{gemm_output_shape, meta.dtype}}});

    syn_out(0) = std::move(
        ReshapeHelper(graph, output[0].get(), meta.shape, self_dtype, 0));
  } else if (
      (self_dim >= 1 && other_dim >= 1) && (self_dim >= 3 || other_dim >= 3)) {
    syn_out(0) = std::move(OpBackend::BuildNode(
        this,
        graph,
        {"batch_gemm",
         {syn_in(0), syn_in(1)},
         {{meta.shape, meta.dtype, 0}}})[0]);
  } else {
    TORCH_CHECK(false, "Not supported matmul configuration.");
  }
}

OutputMetaDataVector MatmulBwdMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 1);
  const auto& other = stack_tensor(stack, 2);

  OutputMetaDataVector meta{
      {self.scalar_type(), self.sizes().vec()},
      {other.scalar_type(), other.sizes().vec()}};
  return meta;
}

MatmulBwd::MatmulBwd(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "matmul_bwd", scalar_type, {1, 2}, {}, {}, false) {
  SetOutputMetaFn(MatmulBwdMeta);
}

} // namespace habana

// When below flag is enabled, aten.linear and aten.matmul decompositions
// are overriden in eager and torch.compile.
static const auto& MatmulKernelRegistry =
    GET_ENV_FLAG_NEW(PT_HPU_OVERRIDE_LINEAR_MATMUL_EAGER)
    ? habana::KernelRegistry()
          .add("hpu::matmul", KERNEL_FN_GLOBAL(habana::Matmul))
          .add("hpu::matmul_bwd", KERNEL_FN_GLOBAL(habana::MatmulBwd))
    : habana::KernelRegistry();
