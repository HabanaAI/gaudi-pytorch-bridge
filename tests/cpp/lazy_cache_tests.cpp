/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_lazy/hpu_lazy_cache.h"
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
  auto las = habana_lazy::LazyArgumentSpec(
      true,
      inputs,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      0);

  // Look for the lazyArgumentSpec in lazy cache
  auto jit_graph =
      habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
          las.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph, nullptr);

  // End the test by clearing the cache for later tests
  habana_lazy::LazyGraphCache::GetLazyCache().Clear();
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
  auto las = habana_lazy::LazyArgumentSpec(
      true,
      inputs,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      0);

  // Look for the LazyArgumentSpec in lazy cache
  auto jit_graph =
      habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
          las.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph, nullptr);

  // Create a JIT IR graph corresponding to the 3 nodes
  auto g = habana_lazy_test::CreateJITGraph();

  // Add the JIR IR against the lazyArgumentSpec in cache
  habana_lazy::LazyGraphCache::GetLazyCache().Add(las.hashCode(), g);

  // This time, the cache lookup should find a cache hit
  jit_graph = habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
      las.hashCode());
  EXPECT_EQ(jit_graph, g);

  // End the test by clearing the cache for later tests
  habana_lazy::LazyGraphCache::GetLazyCache().Clear();
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
  auto las1 = habana_lazy::LazyArgumentSpec(
      true,
      inputs1,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      0);

  // Look for the LazyArgumentSpec in lazy cache
  auto jit_graph =
      habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
          las1.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph, nullptr);

  // Create a JIT IR graph corresponding to the 3 nodes
  auto g = habana_lazy_test::CreateJITGraph();

  // Add the JIR IR against the LazyArgumentSpec in cache
  habana_lazy::LazyGraphCache::GetLazyCache().Add(las1.hashCode(), g);

  // 2 input tensors, different shaped tensors
  auto inputs2_ivalues = habana_lazy_test::CreateInputs({{4, 6}, {4, 6}}, {});
  const at::ArrayRef<torch::jit::IValue> inputs2(inputs2_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and inputs
  auto las2 = habana_lazy::LazyArgumentSpec(
      true,
      inputs2,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      0);

  // Look for the LazyArgumentSpec in lazy cache
  jit_graph = habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
      las2.hashCode());

  // Cache hit is expected
  EXPECT_EQ(jit_graph, g);

  // End the test by clearing the cache for later tests
  habana_lazy::LazyGraphCache::GetLazyCache().Clear();
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
  auto las1 = habana_lazy::LazyArgumentSpec(
      true,
      inputs1,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      0);

  // Look for the LazyArgumentSpec in lazy cache
  auto jit_graph =
      habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
          las1.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph, nullptr);

  // Create a JIT IR graph corresponding to the 3 nodes
  auto g = habana_lazy_test::CreateJITGraph();

  // Add the JIR IR against the LazyArgumentSpec in cache
  habana_lazy::LazyGraphCache::GetLazyCache().Add(las1.hashCode(), g);

  // Create another set of 2 tensors
  auto inputs2_ivalues =
      habana_lazy_test::CreateInputs({{8, 2, 3}, {8, 2, 3}}, {});
  const at::ArrayRef<torch::jit::IValue> inputs2(inputs2_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and new inputs
  auto las2 = habana_lazy::LazyArgumentSpec(
      true,
      inputs2,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      0);

  // Look for the LazyArgumentSpec in lazy cache
  jit_graph = habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
      las2.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph, nullptr);

  // End the test by clearing the cache for later tests
  habana_lazy::LazyGraphCache::GetLazyCache().Clear();
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
  auto las1 = habana_lazy::LazyArgumentSpec(
      true,
      inputs,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      0);

  // Look for the LazyArgumentSpec in lazy cache
  auto jit_graph =
      habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
          las1.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph, nullptr);

  // Create a JIT IR graph corresponding to the 3 nodes
  auto g = habana_lazy_test::CreateJITGraph();

  // Add the JIR IR against the LazyArgumentSpec in cache
  habana_lazy::LazyGraphCache::GetLazyCache().Add(las1.hashCode(), g);

  // Create another post order graph
  auto post_order_struct2 = habana_lazy_test::GetPostOrderNodes(true);
  auto& post_order_nodes_hash2 = post_order_struct2.post_order_nodes_hash;

  // Create an LazyArgumentSpec for the new IR nodes and inputs
  auto las2 = habana_lazy::LazyArgumentSpec(
      true,
      inputs,
      post_order_nodes_hash2,
      {},
      value_input_nodes_map,
      0);

  // Look for the LazyArgumentSpec in lazy cache
  jit_graph = habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
      las2.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph, nullptr);

  // End the test by clearing the cache for later tests
  habana_lazy::LazyGraphCache::GetLazyCache().Clear();
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
  auto las1 = habana_lazy::LazyArgumentSpec(
      true,
      inputs1,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      0);

  // Look for the LazyArgumentSpec in lazy cache
  auto jit_graph =
      habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
          las1.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph, nullptr);

  // Create a JIT IR graph corresponding to the 3 nodes
  auto g = habana_lazy_test::CreateJITGraph();

  // Add the JIR IR against the LazyArgumentSpec in cache
  habana_lazy::LazyGraphCache::GetLazyCache().Add(las1.hashCode(), g);

  // Create another set of 2 tensors
  auto inputs2_ivalues =
      habana_lazy_test::CreateInputs({{2, 3}, {2, 3}}, {3.0});
  const at::ArrayRef<torch::jit::IValue> inputs2(inputs2_ivalues);

  // Create an LazyArgumentSpec for the IR nodes and new inputs
  auto las2 = habana_lazy::LazyArgumentSpec(
      true,
      inputs2,
      post_order_nodes_hash,
      {},
      value_input_nodes_map,
      0);

  // Look for the LazyArgumentSpec in lazy cache
  jit_graph = habana_lazy::LazyGraphCache::GetLazyCache().GetOptimizedJITGraph(
      las2.hashCode());

  // Cache miss is expected
  EXPECT_EQ(jit_graph, nullptr);

  // End the test by clearing the cache for later tests
  habana_lazy::LazyGraphCache::GetLazyCache().Clear();
}
