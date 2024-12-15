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
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "jit_fork/ir/ir_node.h"
#include "jit_fork/ir/type_wrapper.h"

namespace habana_torch::jit {

struct Block {
  friend struct Node;
  friend struct Graph;

  AT_DISALLOW_COPY_AND_ASSIGN(Block);
  TORCH_API Block(Graph* graph_, Node* node_);

  at::ArrayRef<Value*> inputs() {
    return input_->outputs();
  }
  at::ArrayRef<const Value*> inputs() const {
    const auto& inputs = input_->outputs();
    return {inputs.data(), inputs.size()};
  }
  at::ArrayRef<Value*> outputs() {
    return output_->inputs();
  }
  at::ArrayRef<const Value*> outputs() const {
    return static_cast<const Node*>(output_)->inputs();
  }
  graph_node_list nodes() {
    return {input_, kNextDirection};
  }
  const_graph_node_list nodes() const {
    return {input_, kNextDirection};
  }
  Node* return_node() {
    return output_;
  }
  const Node* return_node() const {
    return output_;
  }
  Node* param_node() {
    return input_;
  }
  const Node* param_node() const {
    return input_;
  }
  /**
   * @warning NEVER pass raw pointer of smart pointer managed Graph to Python.
   * Check #87343 for details.
   */
  Graph* owningGraph() {
    return graph_;
  }
  const Graph* owningGraph() const {
    return graph_;
  }
  Node* owningNode() {
    return owning_node_;
  }
  const Node* owningNode() const {
    return owning_node_;
  }

  Value* addInput(const std::string& name = "") {
    Value* v = input_->addOutput();
    v->setDebugName(name);
    return v;
  }
  Value* insertInput(size_t i, const std::string& name = "") {
    Value* v = input_->insertOutput(i);
    v->setDebugName(name);
    return v;
  }
  void eraseInput(size_t i) {
    input_->eraseOutput(i);
  }
  void removeAllInputs() {
    input_->removeAllOutputs();
  }
  size_t registerOutput(Value* v) {
    output_->addInput(v);
    return outputs().size() - 1;
  }
  size_t insertOutput(size_t i, Value* n) {
    output_->insertInput(i, n);
    return i;
  }
  void eraseOutput(size_t i) {
    output_->removeInput(i);
  }
  void removeAllOutputs() {
    output_->removeAllInputs();
  }

  void replaceOutput(size_t i, Value* n) {
    output_->replaceInput(i, n);
  }
  void permuteOutputs(const std::vector<size_t>& new_inputs) {
    output_->permuteInputs(new_inputs);
  }
  void permuteInputs(const std::vector<size_t>& new_inputs) {
    input_->permuteOutputs(new_inputs);
  }

  Node* appendNode(Node* n) {
    HABANA_ASSERT(n->graph_ == graph_ && !n->inBlockList());
    n->insertBefore(output_);
    return n;
  }
  Node* prependNode(Node* n) {
    HABANA_ASSERT(n->graph_ == graph_ && !n->inBlockList());
    n->insertAfter(input_);
    return n;
  }

  // clone all inputs, nodes, and outputs from src and append them
  // to the inputs, nodes, and outputs of this block
  // value_map is used whenever a node in src references a free variable
  // in src to look up its corresponding value
  TORCH_API void cloneFrom(Block* src, std::function<Value*(Value*)> value_map);
  TORCH_API void remapTypes(
      const std::function<TypeWrapper(TypeWrapper)>& type_map);

  TORCH_API std::shared_ptr<Wrap<Block>> wrap() {
    if (!wrap_) {
      wrap_ = std::make_shared<Wrap<Block>>(this);
    }
    return wrap_;
  }

  virtual ~Block() {
    if (wrap_) {
      wrap_->clear();
    }
  }

  void clear() {
    removeAllOutputs();
    for (auto it = nodes().rbegin(); it != nodes().rend(); it++) {
      it.destroyCurrent();
    }
    removeAllInputs();
  }

 private:
  void reIndexTopology();

  // get rid of all nodes
  // destroys in reverse order so that uses internal to this block
  // do not have to be removed before you can destroy the block
  void destroy();

  Graph* const graph_;
  // holds outputs in a way that can be reflected
  // as a Use object
  // also used as the beginning/end of the circular node list to avoid
  // having corner cases where the list is empty.
  Node* const output_;
  Node* const input_;
  Node* const
      owning_node_; // either the node that has this block or nullptr for root
  // a managing wrapper for Python to allow invalidation
  std::shared_ptr<Wrap<Block>> wrap_;
};

} // namespace habana_torch::jit
