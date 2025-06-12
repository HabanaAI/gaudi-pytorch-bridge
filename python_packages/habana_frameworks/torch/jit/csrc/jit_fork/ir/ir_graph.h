/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dynamic_type.h>
#include <ATen/core/enum_type.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/python_stub.h>
#include <torch/csrc/utils/schema_info.h>

namespace habana_torch::jit {

using ::c10::Argument;
using ::c10::FunctionSchema;
using ::c10::IValue;
using ::c10::Symbol;
using ::c10::ivalue::ConstantString;
using ::c10::ivalue::Future;
using ::c10::ivalue::Shared;

class AliasDb;

#define C10_USING(T) using ::c10::T;
C10_FORALL_TYPES(C10_USING)
#undef C10_USING

#define C10_USING(T) using ::c10::T##Ptr;
C10_FORALL_TYPES(C10_USING)
#undef C10_USING

using ::c10::Type;
using ::c10::TypeEnv;
using ::c10::TypePtr;

using ::c10::getTypePtr;
using ::c10::MatchTypeReturn;
using ::c10::TypeKind;

using ::c10::fmap;

struct MatchedSchema;

struct Graph : std::enable_shared_from_this<Graph> {
  AT_DISALLOW_COPY_AND_ASSIGN(Graph);
  friend struct Node;
  friend struct Value;
  friend struct Block;

 private:
  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  std::unordered_set<const Node*> all_nodes;
  std::unordered_set<const Value*> all_values;
  std::unordered_set<const Block*> all_blocks;
  size_t next_unique_{0};

  std::unordered_map<std::string, Value*> unique_names_;
  // name_base_suffix tracks largest suffix currently used by all names sharing
  // same name_base. Key of this map is name_base, value is largest suffix
  // numeric value.
  std::unordered_map<std::string, size_t> name_base_suffix_;

  ScopePtr current_scope_;

  Block* const block_;
  // when insertNode() is called, the node is inserted before this node
  // by default this is set to append to the top level block
  Node* insert_before_;
  int64_t predicted_insert_count_ = 0;

  std::optional<size_t> op_version_;

 public:
  Graph(ScopePtr scope_root = c10::make_intrusive<Scope>())
      : current_scope_(std::move(scope_root)),
        block_(new Block(this, nullptr)),
        insert_before_(return_node()) {}

  at::ArrayRef<Value*> inputs() {
    return block_->inputs();
  }
  at::ArrayRef<const Value*> inputs() const {
    const Block& block = *block_;
    return block.inputs();
  }
  at::ArrayRef<Value*> outputs() {
    return block_->outputs();
  }
  at::ArrayRef<const Value*> outputs() const {
    const Block& block = *block_;
    return block.outputs();
  }
  graph_node_list nodes() {
    return block_->nodes();
  }
  const_graph_node_list nodes() const {
    const Block& block = *block_;
    return block.nodes();
  }
  Node* param_node() {
    return block_->param_node();
  }
  const Node* param_node() const {
    return block_->param_node();
  }
  Node* return_node() {
    return block_->return_node();
  }
  const Node* return_node() const {
    return block_->return_node();
  }
  const std::unordered_map<std::string, Value*>& debugNames() const {
    return unique_names_;
  }

  TORCH_API void push_scope(const std::string& scope_name);
  TORCH_API void pop_scope();

  ScopePtr current_scope() {
    return current_scope_;
  }

  void set_op_version(std::optional<size_t> version) {
    op_version_ = version;
  }

  std::optional<size_t> get_op_version() {
    return op_version_;
  }

  void set_current_scope(ScopePtr scope) {
    current_scope_ = std::move(scope);
  }

  Value* addInput(const std::string& name = "") {
    return block_->addInput(name);
  }
  Value* insertInput(size_t i, const std::string& name = "") {
    return block_->insertInput(i, name);
  }
  void eraseInput(size_t i) {
    block_->eraseInput(i);
  }
  size_t registerOutput(Value* n) {
    return block_->registerOutput(n);
  }
  void eraseOutput(size_t i) {
    block_->eraseOutput(i);
  }

  TORCH_API Node* create(NodeKind kind, size_t num_outputs = 1);
  TORCH_API Node* create(
      NodeKind kind,
      ArrayRef<Value*> inputs,
      size_t num_outputs = 1);

  TORCH_API Node* createNone();
  TORCH_API Node* createAutogradZero();
  TORCH_API Node* createUninitialized(TypePtr typ);
  TORCH_API Node* createWithSubgraph(Symbol kind);
  TORCH_API Node* createDifferentiableSubgraph();
  TORCH_API Node* createTuple(
      at::ArrayRef<Value*> values,
      TupleTypePtr optional_named_tuple = nullptr);
  TORCH_API Node* createTupleUnpack(Value* v);
  TORCH_API Node* createTupleIndex(
      Value* tup,
      Value* idx,
      const TypePtr& output_type);
  TORCH_API Node* createTupleSlice(
      Value* tup,
      int64_t beg,
      int64_t step_size,
      int64_t num_values);
  TORCH_API Node* createEnumName(Value* e);
  TORCH_API Node* createEnumValue(Value* e);
  TORCH_API Node* createList(
      const TypePtr& contained_type,
      at::ArrayRef<Value*> values);
  TORCH_API Node* createListUnpack(Value* v, size_t size);
  // pack outputs of a function following python rules. If there is a single
  // value return a SimpleValue, otherwise pack all the values into a Tuple
  // or List it is dependent on how many different types have "values" that
  // is passed as an argument.
  Value* packValues(
      at::ArrayRef<Value*> values,
      c10::OptNameList field_names = std::nullopt);
  TORCH_API Node* createDict(
      const TypePtr& key_type,
      const TypePtr& value_type,
      at::ArrayRef<Value*> keys,
      at::ArrayRef<Value*> values);
  TORCH_API Node* createNumToTensor(Value* value);
  TORCH_API Node* createObject(const ClassTypePtr& type);
  TORCH_API Node* createSetAttr(
      Value* obj,
      const std::string& field,
      Value* newValue);
  TORCH_API Node* createGetAttr(Value* obj, const std::string& field);
  Value* insertGetAttr(Value* obj, const std::string& field) {
    return insertNode(createGetAttr(obj, field))->output();
  }
  TORCH_API Node* createStore(const std::string& name, Value* v);
  TORCH_API Node* createLoad(const std::string& name, const TypePtr& type);
  TORCH_API Node* createIsInstance(Value* v, at::ArrayRef<TypePtr> types);

  TORCH_API Value* insertUncheckedCast(Value* v, TypePtr type);

  // Insert a ToList operator with argument \p v and output type \p type.
  // \returns the output of the operation.
  TORCH_API Value* insertToList(Value* v, TypePtr type);

  TORCH_API Value* insertFunctionCall(
      torch::jit::Function* callee,
      const MatchedSchema& matched);
  TORCH_API Value* insertMethodCall(
      std::string method_name,
      const MatchedSchema& matched);

  // Note: defined in python_ir.cpp and can be used only in python extension
  Node* createPythonOp(
      THPObjectPtr&& pyobj,
      const std::string& cconv,
      pyobj_list&& scalar_args);
  // clone n, making a new node in _this_ graph.
  // use value_map to translate inputs of n to inputs of the cloned node
  // if copy_blocks is false, it will not recursively clone the nested blocks
  // this node contains.
  TORCH_API Node* createClone(
      Node* n,
      const std::function<Value*(Value*)>& value_map,
      bool copy_blocks = true);

  // Insert constant IValue into the graph.
  TORCH_API Value* insertConstant(
      const IValue& val,
      std::optional<SourceRange> loc = std::nullopt,
      std::optional<ScopePtr> scope = std::nullopt);

  // Schema-driven insert:
  // This inserts a node into the graph with inputs determined from args and
  // kwargs using Python argument matching rules, and checks that the op matches
  // a known schema.
  //
  // If this node successfully completes, it guarentees the node
  // is a correctly-formed invocation of opname
  TORCH_API Value* insert(
      Symbol opname,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs = {},
      const std::optional<SourceRange>& range = {});

  Node* appendNode(Node* n) {
    return block_->appendNode(n);
  }

  Node* prependNode(Node* n) {
    return block_->prependNode(n);
  }

  // insert before insert_before_ node
  // initialized to insert at the end of the top level block
  // can be changed with setInsertPoint()
  Node* insertNode(Node* n) {
    PT_BRIDGE_DEBUG("Inserting new JIT node: ", *n);
    HABANA_ASSERT(
        insert_before_->inBlockList() &&
        "insert point node is no longer in a block list");
    return n->insertBefore(insert_before_);
  }
  // set where nodes are inserted to append to the end of this block
  void setInsertPoint(Block* b) {
    HABANA_ASSERT(b->owningGraph() == this);
    setInsertPoint(b->return_node());
  }
  // set where nodes are inserted to insert _before_ this node
  // for implementation simplicity we only support inserting before a node for
  // now
  void setInsertPoint(Node* n) {
    HABANA_ASSERT(n->owningGraph() == this && n->inBlockList());
    insert_before_ = n;
    predicted_insert_count_ = 0;
  }
  Node* insertPoint() {
    return insert_before_;
  }

  // the top level block
  Block* block() {
    return block_;
  }
  const Block* block() const {
    return block_;
  }

  // Checks well-formedness and invariants of graph
  TORCH_API void lint() const;
  // for use in debugger
  TORCH_API void dump() const;

  TORCH_API ~Graph();

  TORCH_API std::string toString(bool print_source_info = true) const;

  TORCH_API std::ostream& print(
      std::ostream& out,
      bool print_source_info = true) const;

  friend TORCH_API std::ostream& operator<<(std::ostream& out, const Graph& g);

  TORCH_API std::shared_ptr<Graph> copy();
  TORCH_API std::unique_ptr<Graph> copyUnique();
  TORCH_API void remapTypes(const std::function<TypePtr(TypePtr)>& type_map);

 private:
  friend TORCH_API void Lint(const AliasDb* db);
  TORCH_API void freeNode(Node* n);
  TORCH_API void freeValue(Value* v);
  TORCH_API void freeBlock(Block* b);
  void cloneFrom(Graph& src);

  void cloneToUpstreamGraph(
      std::shared_ptr<::torch::jit::Graph>& destination_graph);

  ::torch::jit::Node* createUpstreamClone(
      Node* src_node,
      std::function<::torch::jit::Value*(Value*)>);

 public:
  std::shared_ptr<::torch::jit::Graph> copyToUpstreamGraph();
};
} // namespace habana_torch::jit
