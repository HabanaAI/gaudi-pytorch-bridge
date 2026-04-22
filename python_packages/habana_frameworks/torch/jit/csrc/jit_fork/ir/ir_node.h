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

#include "jit_fork/ir/ir_value.h"
#include "jit_fork/ir/type_wrapper.h"

#include <functional>
#include <iosfwd>
#include <unordered_set>
#include <vector>

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
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/python_stub.h>
#include <torch/csrc/utils/schema_info.h>

namespace habana_torch::jit {

struct Block;

struct Node;
struct Value;
struct Use;
// the list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier
using node_list = std::vector<Node*>;
using value_list = std::vector<Value*>;
using use_list = std::vector<Use>;
template <typename T>
using ArrayRef = at::ArrayRef<T>;
using NodeKind = Symbol;
using topo_position_t = int64_t;
using ::c10::Argument;
using ::c10::FunctionSchema;
using ValueSet = std::unordered_set<const Value*>;

namespace prim {
using namespace ::c10::prim;
}

struct OperatorSet;
template <typename T>
struct OperatorMap;

struct TORCH_API Node {
  AT_DISALLOW_COPY_AND_ASSIGN(Node);
  friend struct Graph;
  friend struct Block;
  friend struct Value;
  friend graph_node_list;
  friend const_graph_node_list;
  friend graph_node_list_iterator;
  friend const_graph_node_list_iterator;

  std::shared_ptr<::torch::jit::Node*> getUpstreamNode();

 private:
  const NodeKind kind_;
  std::vector<Value*> inputs_;
  std::vector<Value*> outputs_;
  // subblocks
  std::vector<Block*> blocks_;
  Graph* graph_;
  Block* owning_block_{nullptr};
  std::optional<SourceRange> source_range_;
  std::optional<std::string> stack_trace_;
  ScopePtr scope_;
  std::optional<InlinedCallStackPtr> callstack_;
  // Assumes FunctionSchemas are persistent, so we don't manage their lifetime.
  // This field is effective a cache that's populated on attribute lookups and
  // invalidated every time we perform an operation that could potentially
  // change the schema. note: mutable because schema_ is effectively a cache
  mutable const torch::jit::Operator* op_{nullptr};
  std::optional<c10::OperatorName> operator_name_{std::nullopt};
  topo_position_t topo_position_{0};
  // a managing wrapper for Python to allow invalidation
  std::shared_ptr<Wrap<Node>> wrap_;
  // Stores the full schema name, if the operator is historic
  // When the operator is deprecated or the name of the operator
  // is changed, we need to rely on this name
  // to retrieve old schemas to successfully apply upgraders
  // for this operator.
  std::optional<std::string> historic_schema_name_ = std::nullopt;

 protected:
  Node(Graph* graph_, NodeKind kind_); // defined after graph
 public:
  // Each Node but Return/Param Nodes are associated with exactly one
  // place in the Node list of the Graph. The Graph itself is a circular
  // doubly-linked list. The Return Node is used as the sentinel for the
  // "beginning"/"end" of the list. This means that you can tell when
  // you've traversed the entire list without means worrying about null
  // pointers. `next_in_graph[0]` is the pointer to the next Node, while
  // `next_in_graph[1]` is the pointer to the previous Node. The
  // linked list is implemented as an array to allow the same iterator
  // class for forward and reversed Node lists. Taken together, this
  // list also represents a topological sort of the Nodes in the Graph.
  Node* next_in_graph[2] = {nullptr, nullptr};

  std::shared_ptr<Wrap<Node>> wrap() {
    if (!wrap_) {
      wrap_ = std::make_shared<Wrap<Node>>(this);
    }
    return wrap_;
  }

  const std::optional<std::string> getHistoricSchemaName() {
    return historic_schema_name_;
  }

  void setHistoricSchemaName(const std::string& name) {
    historic_schema_name_ = name;
  }

  Node*& next() {
    return next_in_graph[kNextDirection];
  }
  Node*& prev() {
    return next_in_graph[kPrevDirection];
  }
  Node* const& next() const {
    return next_in_graph[kNextDirection];
  }
  Node* const& prev() const {
    return next_in_graph[kPrevDirection];
  }

  NodeKind kind() const {
    return kind_;
  }
  Node* setSourceRange(SourceRange r) {
    source_range_ = std::move(r);
    return this;
  }
  SourceRange sourceRange() const;
  Node* setStackTrace(const std::string& stack_trace) {
    stack_trace_ = stack_trace;
    return this;
  }
  std::string stackTrace() const;

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
  Block* owningBlock() {
    return owning_block_;
  }
  const Block* owningBlock() const {
    return owning_block_;
  }
  ScopePtr scope() {
    return scope_;
  }
  void setScope(ScopePtr scope) {
    scope_ = std::move(scope);
  }
  std::string scopeName() const {
    if (!scope_) {
      return "";
    }
    return scope_->namesFromRoot();
  }

  // Copies the source range, scope and callstack from another node.
  Node* copyMetadata(Node* from) {
    this->setSourceRange(from->sourceRange());
    this->setScope(from->scope());
    if (auto cs = from->callstack()) {
      this->setCallStack(*cs);
    }
    return this;
  }

  std::optional<InlinedCallStackPtr> callstack() const {
    return callstack_;
  }
  void setCallStack(InlinedCallStackPtr cs) {
    callstack_ = std::move(cs);
  }

  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  at::ArrayRef<Value*> inputs() {
    return inputs_;
  }
  at::ArrayRef<const Value*> inputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {inputs_.data(), inputs_.size()};
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  at::ArrayRef<Value*> outputs() {
    return outputs_;
  }
  at::ArrayRef<const Value*> outputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {outputs_.data(), outputs_.size()};
  }
  Value* output(size_t i) const {
    return outputs_.at(i);
  }
  bool hasUses() const {
    return std::any_of(outputs_.begin(), outputs_.end(), [](Value* v) {
      return !v->uses().empty();
    });
  }

  void replaceAllUsesWith(Node* n);

  // replaces `this` with a new node with the same inputs and outputs
  // but a new node symbol. does not destroy `this`
  Node* replaceWithNewSymbol(Symbol new_symbol);

  // Checks if this node is dominated by `dominator` which means that
  // `dominator` will always be executed before `this` and `dominator`
  // is in scope of `this.
  bool isDominatedBy(const Node* dominator) const;

  // lots of things like chunk have a single input or single output, so we have
  // a helper to make accessing it easier
  Value* input() {
    HABANA_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  Value* output() {
    HABANA_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  const Value* output() const {
    HABANA_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  const Value* input() const {
    HABANA_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  // Access a particular input.  This is a checked index.
  Value* input(size_t i) const {
    return inputs_.at(i);
  }

  bool hasNamedInput(const std::string& unqualName) const;
  Value* namedInput(const std::string& unqualName) const;
  Value* namedInput(Symbol name) const;

  std::optional<IValue> get(Symbol name) const;

  template <typename T>
  std::optional<T> get(Symbol name) const {
    if (auto v = get(name)) {
      return v->template to<T>();
    }
    return std::nullopt;
  }

  // Returns true if the value of input name is statically known
  bool is_constant(Symbol name) const {
    return static_cast<bool>(get(name));
  }
  bool mustBeNone() const;

  bool isNondeterministic() const;
  bool hasSideEffects() const;

  // instructions lowered by the interpreter and not run in the optimized graph
  bool notExecutedOp() const {
    return kind_ == prim::Constant || kind_ == prim::profile ||
        kind_ == prim::profile_ivalue;
  }

  // Graphs

  // Note [Topological invariant]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // We always maintain an up-to-date topological ordering of all nodes via
  // the next()/prev() links.  All transformations to graphs must preserve
  // this topological ordering: for example, it is only valid to 'addInput'
  // with an input which is topologically before the current node.
  //
  // Usually, it is obvious whether or not topological order is maintained;
  // for example, if you are adding nodes to the end of the topsort, it's
  // impossible for them to refer to inputs that are not in the topsort.
  // If it is not obvious, please comment accordingly.

  // Add 'node' as an input to 'this' at the end of existing
  // arguments.  Returns the added node for ease of chaining.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.addInput(%4)
  // Result:  %3 = f(%1, %2, %4)
  Value* addInput(Value* value);

  // Add 'value' as an input to 'this' at the specified position in the
  // arguments. Returns the added value for ease of chaining.
  Value* insertInput(size_t i, Value* value);

  // Replace the input of 'this' at position 'i' with
  // 'newValue', returning the old node.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.replaceInput(1, %4)
  // Result:  %3 = f(%1, %4)
  Value* replaceInput(size_t i, Value* newValue);

  // Replace all occurrences of 'from' in the inputs of this
  // node with 'to'. Corresponds to llvm's replaceUsesOfWith.
  //
  // Given:   %3 = f(%1, %2, %1)
  // Execute: %3.replaceInputWith(%1, %4)
  // Result:  %3 = f(%4, %2, %4)
  void replaceInputWith(Value* from, Value* to);

  Value* addOutput();

  Value* insertOutput(size_t i);

  void eraseOutput(size_t i);

  Block* addBlock();
  void eraseBlock(size_t i);

  // Each Node can have a list of subblocks. These are used to define structured
  // nested control flow operators such as If and Loop.
  // The meaning of a block is specific to the kind of node it is in, but
  // all blocks share these semantics:
  // * Nested lexical scoping: If a node 'Parent' has a subblock which contains
  //   a node 'Child', Child can use any value that was in scope for the Parent
  //   node in addition to any values defined before 'Child' in the subblock.
  // * The list of inputs to the block are in scope for the duration of the
  //   block
  // * the outputs of the Parent node are not in scope for the subblocks
  // Typically the inputs to a block that represents control flow act as
  // as the equivalents phi-nodes in standard SSA form,
  // defining a new Value to represent any term that has multiple
  // definitions depending on how control flowed. Outputs of the node containing
  // control flow serve a similiar purpose defining new values for variables
  // that would have different definitions depending on which way control
  // flowed.

  at::ArrayRef<Block*> blocks() {
    return blocks_;
  }
  at::ArrayRef<const Block*> blocks() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {blocks_.data(), blocks_.size()};
  }

  // Is 'this' before 'n' in the topological order?
  bool isBefore(const Node* n) const;

  // Is 'this' after 'n' in the topological order?
  bool isAfter(const Node* n) const;

  // Insert unattached 'this' node before 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertBefore(%4)
  // Result:  %3 = f(%1, %2)
  //          %5 = h(%1)
  //          %4 = g(%3)
  Node* insertBefore(Node* n);

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given: %3 = f(%1, %2)
  //        %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertAfter(%4)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%1)
  Node* insertAfter(Node* n);

  // Move 'this' (already in the graph) after 'n' in the topological order.
  //
  // NOTE: Does not check that value dependencies are preserved, see
  //   AliasDb::moveAfterTopologicallyValid
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.moveAfter(%3)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  //
  void moveAfter(Node* n);

  // Move a node 'n' (already in the graph) before 'this' in the topological
  // order.
  //
  // NOTE: Does not check that value dependencies are preserved, see
  //   AliasDb::moveBeforeTopologicallyValid
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %3.moveBefore(%2)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  void moveBefore(Node* n);

  // Remove the input at 'i' from this node.
  //
  // WARNING: This is O(n) in the number of inputs, so avoid repeatedly calling
  // removeInput.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeInput(1)
  // Result: %3 = f(%1)
  void removeInput(size_t i);

  // Remove all inputs from a node.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeAllInputs()
  // Result: %3 = f()
  void removeAllInputs();

  // Remove all outputs from a node.
  //
  // Given: %1, %2 = f()
  // Execute:removeAllInputs()
  // Result: = f()
  void removeAllOutputs();

  // Rearrange the ordering of inputs or outputs of a node
  // Given: %3 = f(%1, %2)
  // Execute: %3.permuteInputs({1, 0})
  // Result: %3 = f(%2, %1)
  // Each index must appear exactly once
  void permuteInputs(const std::vector<size_t>& new_inputs);
  void permuteOutputs(const std::vector<size_t>& new_inputs);

  // iterators of the node list starting at this node
  // useful for resuming a search starting at this node
  graph_node_list_iterator iterator() {
    return {this, 0};
  }
  graph_node_list_iterator reverseIterator() {
    return iterator().reverse();
  }
  const_graph_node_list_iterator iterator() const {
    return {this, 0};
  }
  const_graph_node_list_iterator reverseIterator() const {
    return iterator().reverse();
  }

  // Remove 'this' from the instruction list and deallocate it.
  //
  // Invariant: no outputs of 'this' may have any uses.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.destroy()
  // Result: %3 = g(%1)
  void destroy();

  // Dynamically cast this node to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid..
  //
  // Example usage: if(auto s = n.cast<Select>()) { ... }
  template <typename T>
  T* cast() {
    if (T::Kind == kind()) {
      return static_cast<T*>(this);
    }
    return nullptr;
  }
  template <typename T>
  const T* cast() const {
    if (T::Kind == kind()) {
      return static_cast<const T*>(this);
    }
    return nullptr;
  }

  template <typename T>
  T* expect() {
    HABANA_ASSERT(
        T::Kind == kind(),
        "expected a ",
        T::Kind.toDisplayString(),
        " but found a ",
        kind().toDisplayString());
    return static_cast<T*>(this);
  }

  bool matches(const FunctionSchema& schema) const;

  // XXX: this function is meant to be used with string literals only!
  bool matches(
      const char* signature_literal,
      at::ArrayRef<Symbol> const_inputs = {}) const;

  bool isMemberOf(const OperatorSet& os) const;
  template <typename T>
  bool isMemberOf(const OperatorMap<T>& om) const {
    auto it = om.map.find(kind());
    if (it == om.map.end()) {
      return false;
    }
    for (auto& op : it->second) {
      if (matches(op.first->schema())) {
        return true;
      }
    }
    return false;
  }

  void setOperatorName(const c10::OperatorName& operator_name);
  const c10::OperatorName& getOperatorName() const;

  const FunctionSchema& schema() const;
  const FunctionSchema* maybeSchema() const;
  const torch::jit::Operator& getOperator() const;

  const torch::jit::Operator* maybeOperator() const;

  void dump() const;

  std::ostream& print(
      std::ostream& out,
      size_t level,
      std::vector<const Node*>* groups,
      bool print_source_info = true,
      bool print_attributes = true,
      bool print_scopes = true,
      bool print_body = true) const;

  virtual ~Node() {
    if (wrap_) {
      wrap_->clear();
    }
  }

  // Methods for accessing attributes
  Node* copyAttributes(const Node& rhs) {
    values_.clear();
    for (const AVPtr& i : rhs.values_) {
      values_.push_back(i->clone());
    }
    return this;
  }
  bool hasAttribute(Symbol name) const {
    HABANA_ASSERT(name.is_attr());
    return findAttr(name, false) != values_.end();
  }
  bool hasAttributeS(const std::string& name) const {
    return hasAttribute(Symbol::attr(name));
  }
  AttributeKind kindOf(Symbol name) const {
    HABANA_ASSERT(name.is_attr());
    return (*findAttr(name, true))->kind();
  }
  AttributeKind kindOfS(const std::string& name) const {
    return kindOf(Symbol::attr(name));
  }
  Node* removeAttribute(Symbol name) {
    HABANA_ASSERT(name.is_attr());
    values_.erase(findAttr(name, true));
    return this;
  }
  Node* removeAttributeS(const std::string& name) {
    return removeAttribute(Symbol::attr(name));
  }
  bool hasAttributes() const {
    return !values_.empty();
  }
  size_t numAttributes() const {
    return values_.size();
  }
  // The names are returned in order, since name actually is the index.
  std::vector<Symbol> attributeNames() const {
    std::vector<Symbol> names;
    names.reserve(values_.size());
    for (const AVPtr& a : values_) {
      names.push_back(a->name);
    }
    return names;
  }
  std::vector<const char*> attributeNamesS() const {
    std::vector<const char*> names;
    names.reserve(values_.size());
    for (const AVPtr& a : values_) {
      names.push_back(a->name.toUnqualString());
    }
    return names;
  }

#define CREATE_ACCESSOR(Kind, method)                           \
  Node* method##_(Symbol name, Kind##Attr::ConstructorType v) { \
    return setAttr<Kind##Attr>(                                 \
        name, std::forward<Kind##Attr::ConstructorType>(v));    \
  }                                                             \
  const Kind##Attr::ValueType& method(Symbol name) const {      \
    return getAttr<Kind##Attr>(name);                           \
  }

  CREATE_ACCESSOR(Float, f)
  CREATE_ACCESSOR(Complex, c)
  CREATE_ACCESSOR(Floats, fs)
  CREATE_ACCESSOR(ComplexVals, cs)
  CREATE_ACCESSOR(String, s)
  CREATE_ACCESSOR(Strings, ss)
  CREATE_ACCESSOR(Int, i)
  CREATE_ACCESSOR(Bool, b)
  CREATE_ACCESSOR(Bools, bs)
  CREATE_ACCESSOR(Ints, is)
  CREATE_ACCESSOR(Graph, g)
  CREATE_ACCESSOR(Graphs, gs)
  CREATE_ACCESSOR(Type, ty)
  CREATE_ACCESSOR(Types, tys)
  CREATE_ACCESSOR(IValue, ival)

#undef CREATE_ACCESSOR

  // Our Graphs are not very const-correct, so we need to allow returning
  // non-const references too
  GraphAttr::ValueType& g(Symbol name) {
    return getAttr<GraphAttr>(name);
  }

  // does not use CREATE_ACCESSOR because we need additional asserts
  Node* t_(Symbol name, TensorAttr::ConstructorType v) {
    return setAttr<TensorAttr>(
        name, std::forward<TensorAttr::ConstructorType>(v));
  }
  const TensorAttr::ValueType& t(Symbol name) const {
    return getAttr<TensorAttr>(name);
  }

  Node* ts_(Symbol name, TensorsAttr::ConstructorType v) {
    return setAttr<TensorsAttr>(
        name, std::forward<TensorsAttr::ConstructorType>(v));
  }
  const TensorsAttr::ValueType& ts(Symbol name) const {
    return getAttr<TensorsAttr>(name);
  }

  Block* findCommonAncestorBlockWith(Node* n);

  size_t blocksFromGraphBlock();

 private:
  void printAttrValue(std::ostream& out, const Symbol& name) const;
  void printAttributes(std::ostream& out, bool ignore_subgraph) const;

  template <typename T>
  Node* setAttr(Symbol name, typename T::ConstructorType v) {
    HABANA_ASSERT(name.is_attr());
    auto it = findAttr(name, false);
    auto nv = AVPtr(new T(name, std::forward<typename T::ConstructorType>(v)));
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (it == values_.end()) {
      values_.push_back(std::move(nv));
    } else {
      *it = std::move(nv);
    }
    return this;
  }
  template <typename T>
  typename T::ValueType& getAttr(Symbol name) const {
    HABANA_ASSERT(name.is_attr());
    auto it = findAttr(name, true);
    auto* child = dynamic_cast<T*>(it->get());
    if (child == nullptr) {
      throw IRAttributeError(name, true);
    }
    return child->value();
  }
  using AVPtr = AttributeValue::Ptr;
  // NB: For determinism, we use a vector rather than a hash map.  This does
  // mean that lookups are O(n), so you shouldn't use Attributes to store
  // a big pile of messages.
  std::vector<AVPtr> values_;
  std::vector<AVPtr>::iterator findAttr(Symbol name, bool required) {
    HABANA_ASSERT(name.is_attr());
    auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) {
      return v->name == name;
    });
    if (required && it == values_.end()) {
      throw IRAttributeError(name, false);
    }
    HABANA_ASSERT(!required || it != values_.end());
    return it;
  }
  std::vector<AVPtr>::const_iterator findAttr(Symbol name, bool required)
      const {
    HABANA_ASSERT(name.is_attr());
    auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) {
      return v->name == name;
    });
    if (required && it == values_.end()) {
      throw IRAttributeError(name, false);
    }
    HABANA_ASSERT(!required || it != values_.end());
    return it;
  }

  enum class MoveSide { BEFORE, AFTER };
  bool isBeforeOrAfter(const Node* n, MoveSide moveSide) const;

  std::pair<Value*, const Argument&> findInput(Symbol name);
  // Lookup iterator in use list of _input i_ that corresponds to its use of
  // _this_
  use_list::iterator findUseForInput(size_t i);

  // remove the use of input i, this sets input i to nullptr, but
  // is only used internally to Node before setting it to a new value
  // or erasing the entry from the list.
  Value* dropInput(size_t i);

  bool inBlockList() const {
    if (next() == nullptr) {
      HABANA_ASSERT(prev() == nullptr);
    }
    return next() != nullptr;
  }

  void removeFromList();
  void lint() const;

  void assignTopoPosition();

 protected:
  // subclasses must override
  // this function is used by createClone to initialize a new version
  // of a node in another graph. It should allocate a new instance of the same
  // concrete type as 'this', but in graph 'g' which might be different
  // than graph_
  virtual Node* allocNewInstance(Graph* g) {
    return new Node(g, kind());
  }
  // create a copy of all properties of Node s into this.
  // subclasses should extend if they have additional information to copy.
  // 'this' will be allocated with s->allocNewInstance(g) so it should have
  // the same concrete type as 's'
  virtual void cloneFrom(Node* s);

 private:
  void copyAttributesIntoUpstreamNode(::torch::jit::Node*);
};

TORCH_API std::vector<Node*> findAllNodes(Graph& g, Symbol kind, bool recurse);
TORCH_API std::vector<Node*> findAllNodes(Block& b, Symbol kind, bool recurse);
TORCH_API std::vector<Node*> findAllNodes(
    at::ArrayRef<Block*> a,
    Symbol kind,
    bool recurse);

struct TORCH_API OperatorSet {
  OperatorSet(std::initializer_list<const char*> sig_literals);
  std::vector<std::shared_ptr<torch::jit::Operator>> getOps() const;
  void insert(std::initializer_list<const char*> sig_literals);

 private:
  friend struct Node;
  std::unordered_map<Symbol, std::vector<std::shared_ptr<torch::jit::Operator>>>
      ops;
};

template <typename T>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct OperatorMap {
  // Type aliasing
  using OpMapType =
      typename std::pair<std::shared_ptr<torch::jit::Operator>, T>;
  using ValueType = std::vector<OpMapType>;
  using MapType = std::unordered_map<Symbol, ValueType>;

  OperatorMap() = default;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit OperatorMap(
      std::initializer_list<std::pair<std::shared_ptr<torch::jit::Operator>, T>>
          init) {
    insert(init);
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit OperatorMap(std::initializer_list<std::pair<const char*, T>> init) {
    insert(init);
  }

  void insert(const std::shared_ptr<torch::jit::Operator>& op, T val) {
    // Remove if exists before insert
    erase(op);

    map[Symbol::fromQualString(op->schema().name())].emplace_back(
        std::make_pair(op, val));
  }

  void insert(const OperatorSet& op_set, T val) {
    for (auto& op : op_set.getOps()) {
      insert(op, val);
    }
  }

  void insert(
      std::initializer_list<std::pair<std::shared_ptr<torch::jit::Operator>, T>>
          v) {
    for (auto& el : v) {
      insert(el.first, el.second);
    }
  }

  void insert(std::initializer_list<std::pair<const char*, T>> v) {
    for (auto& el : v) {
      insert(torch::jit::getOperatorForLiteral(el.first), el.second);
    }
  }

  void erase(const std::shared_ptr<torch::jit::Operator>& op) {
    auto it = map.find(Symbol::fromQualString(op->schema().name()));
    if (it == map.end()) {
      return;
    }
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first->schema() == op->schema()) {
        it->second.erase(vit);
        break;
      }
    }
    if (it->second.empty()) {
      map.erase(Symbol::fromQualString(op->schema().name()));
    }
  }

  bool contains(const torch::jit::Operator& op) const {
    const auto it = map.find(Symbol::fromQualString(op.schema().name()));
    if (it == map.end()) {
      return false;
    }
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first->schema() == op.schema()) {
        return true;
      }
    }
    return false;
  }

  bool contains(const Node* n) const {
    return n->maybeOperator() && contains(n->getOperator());
  }

  std::optional<T> find(const torch::jit::Operator& op) {
    const auto it = map.find(Symbol::fromQualString(op.schema().name()));
    if (it == map.end()) {
      return std::nullopt;
    }
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first->schema() == op.schema()) {
        return vit->second;
      }
    }
    return std::nullopt;
  }

  // TODO: return iterator
  std::vector<OpMapType> getAllKeysAndValues() const {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<OpMapType> keys_values;
    for (auto& symbol_mapping : map) {
      auto& vec = symbol_mapping.second;
      for (auto& pair : vec) {
        keys_values.push_back(pair);
      }
    }
    return keys_values;
  }

 private:
  friend struct Node;
  MapType map;
};

} // namespace habana_torch::jit
