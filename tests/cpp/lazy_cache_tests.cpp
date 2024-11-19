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

#include "backend/jit_graph_cache.h"
#include "habana_lazy/lazy_arg_spec.h"
#include "habana_lazy_test_infra.h"

#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>
#include <stdexcept>

/*
 * Cache miss for empty cache
 */
TEST(LazyCacheTest, CacheMissEmptyCache) {
  // 3 Node vector from first level IR
  auto post_order_struct = habana_lazy_test::GetPostOrderNodes();
  auto& post_order_nodes_hash = post_order_struct.post_order_nodes_hash;
  habana_lazy::ir::ValueNodeListMap value_input_nodes_map;

  // 2 input tensors
  auto input_ivalues = habana_lazy_test::CreateInputs({{2, 3}, {2, 3}}, {});
  const at::ArrayRef<torch::jit::IValue> inputs(input_ivalues);

  // Create an lazyArgumentSpec for the IR nodes and inputs
  std::vector<size_t> parent_vec(inputs.size(), ULONG_MAX);
  auto las = habana_lazy::LazyArgumentSpec(
      true,
      inputs,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      {},
      parent_vec);

  // Look for the lazyArgumentSpec in lazy cache
  auto jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph_and_meta_data, nullptr);

  // End the test by clearing the cache for later tests
  habana::JitGraphCache::GetJitCache().Clear();
}

/*
 * Cache hit for:
 * - same post order input nodes
 * - same input tensors
 */
TEST(LazyCacheTest, CacheHitSameInput) {
  // 3 Node vector from first level IR
  auto post_order_struct = habana_lazy_test::GetPostOrderNodes();
  auto& post_order_nodes_hash = post_order_struct.post_order_nodes_hash;
  habana_lazy::ir::ValueNodeListMap value_input_nodes_map;

  // 2 input tensors
  auto input_ivalues = habana_lazy_test::CreateInputs({{2, 3}, {2, 3}}, {});
  const at::ArrayRef<torch::jit::IValue> inputs(input_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and inputs
  std::vector<size_t> parent_vec(inputs.size(), ULONG_MAX);
  auto las = habana_lazy::LazyArgumentSpec(
      true,
      inputs,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      {},
      parent_vec);

  // Look for the LazyArgumentSpec in lazy cache
  auto jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph_and_meta_data, nullptr);

  // Create a JIT IR graph corresponding to the 3 nodes
  auto g = habana_lazy_test::CreateJITGraph();
  auto g_and_m_data =
      std::make_shared<habana::OptimizedJITGraphAndMetaData>(g, inputs);

  // Add the JIR IR against the lazyArgumentSpec in cache
  habana::JitGraphCache::GetJitCache().Add(las.hashCode(), g_and_m_data);

  // This time, the cache lookup should find a cache hit
  jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las.hashCode());
  EXPECT_EQ(jit_graph_and_meta_data, g_and_m_data);

  // End the test by clearing the cache for later tests
  habana::JitGraphCache::GetJitCache().Clear();
}

/*
 * Cache hit for:
 * - same post order input nodes
 * - same number of input tensors, each tensor has
 *   same dimension but different shapes
 */
TEST(LazyCacheTest, CacheHitSameDimTensors) {
  // 3 Node vector from first level IR
  auto post_order_struct = habana_lazy_test::GetPostOrderNodes();
  auto& post_order_nodes_hash = post_order_struct.post_order_nodes_hash;
  habana_lazy::ir::ValueNodeListMap value_input_nodes_map;

  // 2 input tensors
  auto inputs1_ivalues = habana_lazy_test::CreateInputs({{2, 3}, {2, 3}}, {});
  const at::ArrayRef<torch::jit::IValue> inputs1(inputs1_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and inputs
  std::vector<size_t> parent_vec1(inputs1.size(), ULONG_MAX);
  auto las1 = habana_lazy::LazyArgumentSpec(
      true,
      inputs1,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      {},
      parent_vec1);

  // Look for the LazyArgumentSpec in lazy cache
  auto jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las1.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph_and_meta_data, nullptr);

  // Create a JIT IR graph corresponding to the 3 nodes
  auto g = habana_lazy_test::CreateJITGraph();
  auto g_and_m_data =
      std::make_shared<habana::OptimizedJITGraphAndMetaData>(g, inputs1);

  // Add the JIR IR against the LazyArgumentSpec in cache
  habana::JitGraphCache::GetJitCache().Add(las1.hashCode(), g_and_m_data);

  // 2 input tensors, different shaped tensors
  auto inputs2_ivalues = habana_lazy_test::CreateInputs({{4, 6}, {4, 6}}, {});
  const at::ArrayRef<torch::jit::IValue> inputs2(inputs2_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and inputs
  std::vector<size_t> parent_vec2(inputs2.size(), ULONG_MAX);
  auto las2 = habana_lazy::LazyArgumentSpec(
      true,
      inputs2,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      {},
      parent_vec2);

  // Look for the LazyArgumentSpec in lazy cache
  jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las2.hashCode());

  // Cache hit is expected
  EXPECT_EQ(jit_graph_and_meta_data, g_and_m_data);

  // End the test by clearing the cache for later tests
  habana::JitGraphCache::GetJitCache().Clear();
}

/*
 * Cache miss for:
 * - same post order input nodes
 * - same number of input tensors, each tensor has
 *   different dimensions
 */
TEST(LazyCacheTest, CacheMissDiffInputs) {
  // 3 Node vector from first level IR
  auto post_order_struct = habana_lazy_test::GetPostOrderNodes();
  auto& post_order_nodes_hash = post_order_struct.post_order_nodes_hash;
  habana_lazy::ir::ValueNodeListMap value_input_nodes_map;

  // 2 input tensors
  auto inputs1_ivalues = habana_lazy_test::CreateInputs({{2, 3}, {2, 3}}, {});
  const at::ArrayRef<torch::jit::IValue> inputs1(inputs1_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and inputs
  std::vector<size_t> parent_vec1(inputs1.size(), ULONG_MAX);
  auto las1 = habana_lazy::LazyArgumentSpec(
      true,
      inputs1,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      {},
      parent_vec1);

  // Look for the LazyArgumentSpec in lazy cache
  auto jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las1.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph_and_meta_data, nullptr);

  // Create a JIT IR graph corresponding to the 3 nodes
  auto g = habana_lazy_test::CreateJITGraph();
  auto g_and_m_data =
      std::make_shared<habana::OptimizedJITGraphAndMetaData>(g, inputs1);

  // Add the JIR IR against the LazyArgumentSpec in cache
  habana::JitGraphCache::GetJitCache().Add(las1.hashCode(), g_and_m_data);

  // Create another set of 2 tensors
  auto inputs2_ivalues =
      habana_lazy_test::CreateInputs({{8, 2, 3}, {8, 2, 3}}, {});
  const at::ArrayRef<torch::jit::IValue> inputs2(inputs2_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and new inputs
  std::vector<size_t> parent_vec2(inputs2.size(), ULONG_MAX);
  auto las2 = habana_lazy::LazyArgumentSpec(
      true,
      inputs2,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      {},
      parent_vec2);

  // Look for the LazyArgumentSpec in lazy cache
  jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las2.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph_and_meta_data, nullptr);

  // End the test by clearing the cache for later tests
  habana::JitGraphCache::GetJitCache().Clear();
}

/*
 * Cache miss for:
 * - different post order input nodes
 * - same input tensors
 */
TEST(LazyCacheTest, CacheMissDiffGraph) {
  // 3 Node vector from first level IR
  auto post_order_struct = habana_lazy_test::GetPostOrderNodes();
  auto& post_order_nodes_hash = post_order_struct.post_order_nodes_hash;
  habana_lazy::ir::ValueNodeListMap value_input_nodes_map;

  // 2 input tensors
  auto inputs_ivalues = habana_lazy_test::CreateInputs({{2, 3}, {2, 3}}, {});
  const at::ArrayRef<torch::jit::IValue> inputs(inputs_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and inputs
  std::vector<size_t> parent_vec(inputs.size(), ULONG_MAX);
  auto las1 = habana_lazy::LazyArgumentSpec(
      true,
      inputs,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      {},
      parent_vec);

  // Look for the LazyArgumentSpec in lazy cache
  auto jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las1.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph_and_meta_data, nullptr);

  // Create a JIT IR graph corresponding to the 3 nodes
  auto g = habana_lazy_test::CreateJITGraph();
  auto g_and_m_data =
      std::make_shared<habana::OptimizedJITGraphAndMetaData>(g, inputs);

  // Add the JIR IR against the LazyArgumentSpec in cache
  habana::JitGraphCache::GetJitCache().Add(las1.hashCode(), g_and_m_data);

  // Create another post order graph
  auto post_order_struct2 = habana_lazy_test::GetPostOrderNodes(true);
  auto& post_order_nodes_hash2 = post_order_struct2.post_order_nodes_hash;

  // Create an LazyArgumentSpec for the new IR nodes and inputs
  std::vector<size_t> parent_vec2(inputs.size(), ULONG_MAX);
  auto las2 = habana_lazy::LazyArgumentSpec(
      true,
      inputs,
      post_order_nodes_hash2,
      {},
      value_input_nodes_map,
      {},
      parent_vec2);

  // Look for the LazyArgumentSpec in lazy cache
  jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las2.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph_and_meta_data, nullptr);

  // End the test by clearing the cache for later tests
  habana::JitGraphCache::GetJitCache().Clear();
}

/*
 * Cache miss for:
 * - same post order input nodes
 * - same number of input tensors
 * - different scalars as inputs
 * This test is disbled as different scalars are
 * going to be constant nodes within the graph and
 * not graph inputs.
 */
TEST(LazyCacheTest, DISABLED_CacheMissDiffScalars) {
  // 3 Node vector from first level IR
  auto post_order_struct = habana_lazy_test::GetPostOrderNodes(true);
  auto& post_order_nodes_hash = post_order_struct.post_order_nodes_hash;
  habana_lazy::ir::ValueNodeListMap value_input_nodes_map;

  // 2 input tensors
  auto inputs1_ivalues =
      habana_lazy_test::CreateInputs({{2, 3}, {2, 3}}, {2.0});
  const at::ArrayRef<torch::jit::IValue> inputs1(inputs1_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and inputs
  std::vector<size_t> parent_vec1(inputs1.size(), ULONG_MAX);
  auto las1 = habana_lazy::LazyArgumentSpec(
      true,
      inputs1,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      {},
      parent_vec1);

  // Look for the LazyArgumentSpec in lazy cache
  auto jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las1.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph_and_meta_data, nullptr);

  // Create a JIT IR graph corresponding to the 3 nodes
  auto g = habana_lazy_test::CreateJITGraph();
  auto g_and_m_data =
      std::make_shared<habana::OptimizedJITGraphAndMetaData>(g, inputs1);

  // Add the JIR IR against the LazyArgumentSpec in cache
  habana::JitGraphCache::GetJitCache().Add(las1.hashCode(), g_and_m_data);

  // Create another set of 2 tensors
  auto inputs2_ivalues =
      habana_lazy_test::CreateInputs({{2, 3}, {2, 3}}, {3.0});
  const at::ArrayRef<torch::jit::IValue> inputs2(inputs2_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and new inputs
  std::vector<size_t> parent_vec2(inputs2.size(), ULONG_MAX);
  auto las2 = habana_lazy::LazyArgumentSpec(
      true,
      inputs2,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      {},
      parent_vec2);

  // Look for the LazyArgumentSpec in lazy cache
  jit_graph_and_meta_data =
      habana::JitGraphCache::GetJitCache().GetOptimizedJITGraphAndMetaData(
          las2.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph_and_meta_data, nullptr);

  // End the test by clearing the cache for later tests
  habana::JitGraphCache::GetJitCache().Clear();
}

TEST(LazyCacheTest, PadOpCacheTest) {
  torch::Tensor tensor = torch::randn({9, 9, 13, 11});
  torch::Tensor tensorHabana = tensor.to(torch::kHPU);

  namespace F = torch::nn::functional;
  auto outHabana = F::pad(
      tensorHabana, F::PadFuncOptions({-1, -1, -1, -1}).mode(torch::kConstant));
  auto out = F::pad(
      tensor, F::PadFuncOptions({-1, -1, -1, -1}).mode(torch::kConstant));
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);

  auto outHabana2 = F::pad(
      tensorHabana, F::PadFuncOptions({-2, -2, -2, -2}).mode(torch::kConstant));
  auto out2 = F::pad(
      tensor, F::PadFuncOptions({-2, -2, -2, -2}).mode(torch::kConstant));
  equal = out2.allclose(outHabana2.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST(LazyCacheTest, CumsumOpCacheTest) {
  torch::Tensor tensor = torch::randn({9, 9, 13, 11});
  torch::Tensor tensorHabana = tensor.to(torch::kHPU);

  int64_t dim1 = 1;
  auto outHabana = torch::cumsum(tensorHabana, dim1);
  auto out = torch::cumsum(tensor, dim1);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);

  int64_t dim2 = -1;
  auto outHabana2 = torch::cumsum(tensorHabana, dim2);
  auto out2 = torch::cumsum(tensor, dim2);
  equal = out2.allclose(outHabana2.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}
