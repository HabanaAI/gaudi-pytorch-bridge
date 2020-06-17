/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_fuser.h"
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include "habana_bridge/passes/mark_ops/whitelist_ops.h"
#include "habana_helpers/logging.h"

namespace torch {
namespace jit {
namespace habana {

struct HabanaGraphFuser {
  using FusionCallback = std::function<bool(Node*)>;

  Block* block_;
  std::unique_ptr<AliasDb> aliasDb_;
  std::shared_ptr<Graph> graph_;
  FusionCallback callback_ = [&](Node* n) { return isFusableOp(n); };
  Symbol kind_;

  HabanaGraphFuser(Block* block, std::shared_ptr<Graph> graph, Symbol kind)
      : block_(block), graph_(std::move(graph)), kind_(kind) {
    aliasDb_ = std::unique_ptr<AliasDb>(new AliasDb(graph_));
  }

  HabanaGraphFuser(
      Block* block,
      std::shared_ptr<Graph> graph,
      FusionCallback callback,
      Symbol kind)
      : block_(block),
        graph_(std::move(graph)),
        callback_(std::move(callback)),
        kind_(kind) {}

  bool isFusable(Node* node) {
    return callback_(node);
  }

  bool isFusableOp(Node* node) {
    if (node->owningBlock() != block_) {
      return false;
    }

    // Looking up the Op to see if it is whitelisted
    return HabanaWhiteList::is_op_habana_whitelisted(
        node->kind().toQualString());
  }

  std::shared_ptr<Graph> getSubgraph(Node* n) {
    return n->g(attr::Subgraph);
  }

  void mergeFusionGroups(Node* consumer_group, Node* producer_group) {
    // Now we have two fusion groups!
    // Revert the fusion - place all inner nodes of producer back in the outer
    // graph.
    std::vector<Node*> temporary_nodes;
    auto producer_subgraph = getSubgraph(producer_group);

    // Initialize a map of inner graph values to outer graph values
    std::unordered_map<Value*, Value*> inner_to_outer;
    auto inner_inputs = producer_subgraph->inputs();
    auto outer_inputs = producer_group->inputs();
    for (size_t i = 0; i < inner_inputs.size(); ++i) {
      inner_to_outer[inner_inputs[i]] = outer_inputs[i];
    }

    // Clone all nodes
    for (auto inner : producer_subgraph->nodes()) {
      Node* outer = block_->owningGraph()->createClone(
          inner, [&](Value* k) -> Value* { return inner_to_outer.at(k); });
      outer->insertBefore(producer_group);
      temporary_nodes.emplace_back(outer);
      auto inner_outputs = inner->outputs();
      auto outer_outputs = outer->outputs();
      for (size_t i = 0; i < inner_outputs.size(); ++i) {
        inner_to_outer[inner_outputs[i]] = outer_outputs[i];
      }
    }

    // Replace uses of producer_group outputs and destroy the producer
    auto subgraph_outputs = producer_subgraph->outputs();
    for (size_t i = 0; i < subgraph_outputs.size(); ++i) {
      auto outer_output = inner_to_outer.at(subgraph_outputs[i]);
      producer_group->outputs()[i]->replaceAllUsesWith(outer_output);
    }
    producer_group->destroy();
    producer_group =
        nullptr; // Just to get a clear error in case someone uses it

    // Inline the temporary nodes into the first group
    // auto consumer_subgraph = &getSubgraph(consumer_group);
    auto consumer_subgraph = getSubgraph(consumer_group);
    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
         ++it) {
      Node* node = *it;
      SubgraphUtils::mergeNodeIntoSubgraph(node, consumer_group);
      // If any of the outputs are still used then we need to add them
      auto outputs = node->outputs();
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i];
        if (output->uses().empty()) {
          continue;
        }
        consumer_subgraph->registerOutput(consumer_group->outputs()[i]);
        auto new_output = consumer_group->addOutput();
        output->replaceAllUsesWith(new_output);
        new_output->setType(output->type());
      }
      node = nullptr; // node->destroy() is a better alternative, but errors out
                      // due to some internal constraint check in pytorch. Since
                      // it is a temp var, this should also be ok
    }
  }

  at::optional<Node*> tryFuse(Node* consumer, Value* producer) {
    // Check if incoming producer node is fusable and if adding the producer
    // will result in cycles in the graph
    bool shouldFuse = isFusable(consumer) && isFusable(producer->node()) &&
        aliasDb_->moveBeforeTopologicallyValid(producer->node(), consumer);

    if (!shouldFuse) {
      return at::nullopt;
    }
    auto group = consumer;
    if (consumer->kind() != kind_) {
      group = SubgraphUtils::createSingletonSubgraph(consumer, kind_);
    }

    if (producer->node()->kind() == kind_) {
      mergeFusionGroups(group, producer->node());
      return group;
    }

    SubgraphUtils::mergeNodeIntoSubgraph(producer->node(), group);
    return group;
  }

  value_list sortReverseTopological(ArrayRef<Value*> inputs) {
    value_list result;
    for (auto i : inputs) {
      if (i->node()->owningBlock() == block_) {
        result.push_back(i);
      }
    }
    // Sort in reverse topological order
    std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
      return a->node()->isAfter(b->node());
    });
    return result;
  }

  bool usedOnlyInSize(Value* v) {
    const auto& uses = v->uses();
    return std::all_of(uses.begin(), uses.end(), [](const Use& u) {
      return u.user->matches("aten::size(Tensor self) -> int[]");
    });
  }

  Value* broadcastSizes(at::ArrayRef<Value*> sizes) {
    AT_ASSERT(!sizes.empty());
    Graph* graph = sizes[0]->owningGraph();
    Node* broadcast_n =
        graph->insertNode(graph->create(prim::BroadcastSizes, sizes));
    broadcast_n->output()->setType(ListType::ofInts());
    return broadcast_n->output();
  }

  // Builds up expressions that compute shapes of all intermediates (and
  // outputs) of the fusion group, based on the sizes of inputs. You should run
  // DCE to remove those that you end up not using.
  std::unordered_map<Value*, Value*> buildShapeExpressions(Node* fusion_group) {
    WithInsertPoint insert_guard{fusion_group->next()};
    std::unordered_map<Value*, Value*> shape_of;

    Graph* graph = fusion_group->owningGraph();
    auto subgraph = fusion_group->g(attr::Subgraph);

    auto inputs = fusion_group->inputs();
    auto sinputs = subgraph->inputs();
    AT_ASSERT(inputs.size() == sinputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i]->type()->isSubtypeOf(TensorType::get())) {
        shape_of[sinputs[i]] = graph->insert(aten::size, {inputs[i]});
      }
    }

    // When we have a guarantee that an output won't be removed, because it's
    // used in expressions that don't involve size checks, we can use its size
    // instead of computing a long chain of broadcasts, starting from the
    // beginning of the kernel.
    auto outputs = fusion_group->outputs();
    auto soutputs = subgraph->outputs();
    AT_ASSERT(outputs.size() == soutputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (usedOnlyInSize(outputs[i])) {
        continue;
      }
      shape_of[soutputs[i]] = graph->insert(aten::size, {outputs[i]});
    }

    for (Node* n : subgraph->nodes()) {
      // XXX: Use of shape_of.emplace is crucial to the output shape
      // optimization!
      if (n->kind() == prim::FusedConcat) {
        // This is a bit more involved, because we have to account for the case
        // when inputs have different shapes, but fortunately those tensors are
        // always outputs, and so we can simply avoid replacing their queries,
        // because it won't help us.
        continue;
      }
      if (n->kind() == prim::Constant) {
        continue;
      }
      if (n->kind() == prim::ConstantChunk) {
        Node* sizes_node = graph->insertNode(
            graph->create(prim::ChunkSizes, shape_of.at(n->input()), 2));
        sizes_node->i_(attr::dim, n->i(attr::dim));
        sizes_node->i_(attr::chunks, n->i(attr::chunks));
        Value* regular_size = sizes_node->outputs().at(0);
        Value* last_size = sizes_node->outputs().at(1);
        regular_size->setType(ListType::ofInts());
        last_size->setType(ListType::ofInts());
        auto outputs = n->outputs();
        for (Value* o : outputs.slice(0, outputs.size() - 1)) {
          shape_of.emplace(o, regular_size);
        }
        shape_of.emplace(outputs.at(outputs.size() - 1), last_size);
        continue;
      }
      auto tensor_inputs = filter(n->inputs(), [](Value* v) {
        return v->type()->isSubtypeOf(TensorType::get());
      });
      auto shapes =
          fmap(tensor_inputs, [&](Value* v) { return shape_of.at(v); });
      AT_ASSERT(!shapes.empty());
      shape_of.emplace(
          n->output(), shapes.size() == 1 ? shapes[0] : broadcastSizes(shapes));
    }
    return shape_of;
  }

  void removeOutputsUsedOnlyInSize(Node* fusion_group) {
    if (fusion_group->kind() != kind_) {
      return;
    }
    auto subgraph = fusion_group->g(attr::Subgraph);

    auto shape_of = buildShapeExpressions(fusion_group);
    auto outputs = fusion_group->outputs().vec();
    auto soutputs = subgraph->outputs().vec();
    for (int64_t i = static_cast<int64_t>(outputs.size()) - 1; i >= 0; --i) {
      auto output = outputs[i];
      auto soutput = soutputs[i];
      if (usedOnlyInSize(output) && shape_of.count(soutput) > 0) {
        auto uses = output->uses();
        for (Use u : uses) {
          AT_ASSERT(u.user->matches("aten::size(Tensor self) -> int[]"));
          u.user->output()->replaceAllUsesWith(shape_of.at(soutput));
          u.user->destroy();
        }
        fusion_group->eraseOutput(i);
        subgraph->eraseOutput(i);
      }
    }
  }

  void optimizeFusedGraphs() {
    for (Node* node : block_->nodes()) {
      if (node->kind() != kind_) {
        continue;
      }
      auto subgraph = node->g(attr::Subgraph);
      EliminateDeadCode(subgraph);
      EliminateCommonSubexpression(subgraph);
      ConstantPooling(subgraph);
    }
  }

  // returns where to continue scanning, and whether any fusion was made
  std::pair<graph_node_list::iterator, bool> scanNode(Node* consumer) {
    // handle inputs in reverse topological order as well...
    // otherwise in f(a,a+b) it will appear a is used twice if we consider
    // the f-a fusion before the f-(a+b) fusion first.
    auto inputs = sortReverseTopological(consumer->inputs());
    for (auto producer : inputs) {
      auto fusion_group = tryFuse(consumer, producer);
      if (fusion_group) {
        // after fusion, consumer moves into a FusionGroup, so inputs is no
        // longer valid so we rescan the new FusionGroup for more fusions...
        return std::make_pair(fusion_group.value()->reverseIterator(), true);
      }
    }
    return std::make_pair(++consumer->reverseIterator(), false);
  }

  void run() {
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
        bool changed;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    optimizeFusedGraphs();

    // Remove outputs that have been added only because we need their size
    /*for (Node* n : block_->nodes()) {
      removeOutputsUsedOnlyInSize(n);
    }*/

    for (Node* node : block_->nodes()) {
      for (Block* sub_block : node->blocks()) {
        HabanaGraphFuser(sub_block, graph_, kind_).run();
      }
    }
  }

  void PeepholeOptimizeShapeExpressions(Block* block) {
    auto nodes = block->nodes();
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
      Node* node = *it;
      for (Block* subblock : node->blocks()) {
        PeepholeOptimizeShapeExpressions(subblock);
      }
      if (node->kind() == prim::BroadcastSizes) {
        // Remove no-op broadcasts.
        if (node->inputs().size() == 1) {
          node->output()->replaceAllUsesWith(node->input());
          it.destroyCurrent();
          continue;
        }
        // Deduplicate inputs, but use their unique() values to ensure
        // this process only depends on the graph.
        std::map<size_t, Value*> unique_to_value;
        for (Value* input : node->inputs()) {
          unique_to_value.emplace(input->unique(), input);
        }
        if (unique_to_value.size() != node->inputs().size()) {
          std::vector<Value*> inputs;
          inputs.reserve(unique_to_value.size());
          for (auto& entry : unique_to_value) {
            inputs.push_back(entry.second);
          }
          if (inputs.size() == 1) {
            node->output()->replaceAllUsesWith(inputs[0]);
          } else {
            WithInsertPoint insert_guard{node};
            node->output()->replaceAllUsesWith(broadcastSizes(inputs));
          }
          it.destroyCurrent();
          --it; // Revisit the node with deduplicated inputs
          continue;
        }
        // Remove compose simple chains of broadcasts into a single node.
        const auto& uses = node->output()->uses();
        if (uses.size() == 1 && uses[0].user->kind() == prim::BroadcastSizes) {
          Node* user = uses[0].user;
          user->removeInput(uses[0].offset);
          // NB: we don't care about deduplication in here, as we will visit
          // user later.
          for (Value* i : node->inputs()) {
            user->addInput(i);
          }
          it.destroyCurrent();
        }
      }
    }
  }
};

} // namespace habana

void HabanaFuseGraph(std::shared_ptr<torch::jit::Graph>& graph) {
  PT_BRIDGE_BEGIN;
  // First call HPU graph fuser to fuse ops for HPU
  torch::jit::Symbol kind = getHabanaFusedOpSymbol();
  auto g = habana::HabanaGraphFuser(graph->block(), graph, kind);
  g.run();
  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);
  g.PeepholeOptimizeShapeExpressions(graph->block());
  PT_BRIDGE_END;
}

Symbol getHabanaFusedOpSymbol() {
  Symbol habanafusedop_sym =
      torch::jit::Symbol::fromQualString("prim::HabanaFusedOp");
  return habanafusedop_sym;
}

} // namespace jit
} // namespace torch
