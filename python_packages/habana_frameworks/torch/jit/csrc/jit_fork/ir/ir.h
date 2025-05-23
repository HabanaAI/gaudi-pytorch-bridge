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

#include <functional>
#include <iosfwd>
#include <unordered_set>
#include <vector>

#include <fmt/ostream.h>

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

#include "habana_helpers/logging.h"
#include "jit_fork/ir/attributes.h"
#include "jit_fork/ir/graph_node_list.h"
#include "jit_fork/ir/named_value.h"
#include "jit_fork/ir/scope.h"
#include "jit_fork/ir/type_wrapper.h"

// temporary for graphs conversion
#include <torch/csrc/jit/ir/ir.h>

#include "ir_block.h"
#include "ir_graph.h"
#include "ir_node.h"
#include "ir_value.h"

// Forward declare, the real meat is in python_ir.cpp
// todo: caution - it is declared in torch::jit - upstream version, correct it?
template <class T>
class THPPointer;
using THPObjectPtr = THPPointer<PyObject>;
using pyobj_list = std::vector<THPObjectPtr>;

namespace torch {
namespace jit {

struct Function;
struct GraphFunction;

} // namespace jit
} // namespace torch

namespace habana_torch {
namespace jit {

namespace utils {
TORCH_API std::string getNodesModuleHierarchy(const Node& n);
std::string getHeader(const Node* node);
} // namespace utils
class AliasDb;

using ::c10::Argument;
using ::c10::FunctionSchema;
using ::c10::Symbol;

using ::c10::ivalue::Shared;

using ::c10::IValue;
using ::c10::ivalue::Future;

using ::c10::ivalue::ConstantString;

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

namespace prim {
using namespace ::c10::prim;
}
namespace attr {
using namespace ::c10::attr;
}
namespace aten {
using namespace ::c10::aten;
}
namespace cuda {
#if !defined(USE_ROCM)
using namespace ::c10::cuda;
#endif
} // namespace cuda

struct MatchedSchema;

// A Graph represents one "function" of computation.
// It uses a simple ownership model where the graph owns all the nodes inside
// it. All references inside the graph are raw pointers. Destroying the Graph
// will invalidate any pointers to nodes in the graph.
struct Graph;

// Node is the base class of the IR graph. It represents one computation
// and dependencies on a list of Values. The "prim-ops", so to speak.
struct Node;

// A Value represents an input or output to node that is either a
// Tensor or an opaque Handle object, as determined by type().
struct Value;

TORCH_API std::ostream& operator<<(std::ostream& out, const Graph& g);
TORCH_API std::ostream& operator<<(std::ostream& out, const Node& n);

// A list of nodes, with inputs and outputs
struct Block;

struct OperatorSet;
template <typename T>
struct OperatorMap;

/** \brief An utility class for setting temporary insertion points.
 *
 * When an object of this class is created, it stores the current insertion
 * point, sets the new one, and restores the original insertion point when the
 * object is destroyed.
 */
struct WithInsertPoint {
  WithInsertPoint(Node* n) : prev_(n->owningGraph()->insertPoint()) {
    n->owningGraph()->setInsertPoint(n);
  }
  WithInsertPoint(Block* b) : WithInsertPoint(b->return_node()) {}

  ~WithInsertPoint() {
    prev_->owningGraph()->setInsertPoint(prev_);
  }

 private:
  Node* prev_;
};

/** \brief An utility class for setting temporary scopes.
 *
 * When an object of this class is created, it stores the current scope, sets
 * the new one, and restores the original scope when the object is destroyed.
 */
struct WithCurrentScope {
  WithCurrentScope(Graph& g, ScopePtr scope)
      : graph_(&g), prev_scope_(g.current_scope()) {
    g.set_current_scope(std::move(scope));
  }
  ~WithCurrentScope() {
    graph_->set_current_scope(prev_scope_);
  }

 private:
  Graph* graph_;
  ScopePtr prev_scope_;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
inline Value::Value(Node* node_, size_t offset_)
    : node_(node_),
      offset_(offset_),
      unique_(node_->graph_->next_unique_++),
      type_(TensorType::get()) {
  node_->graph_->all_values.emplace(this);
}

inline Value* Value::setType(TypePtr type) {
  AT_ASSERT(type);
  if (auto dyn = type->castRaw<c10::DynamicType>()) {
    type = dyn->fallback();
  }
  type_ = std::move(type);
  for (Use& use : uses_) {
    use.user->op_ = nullptr;
  }
  return this;
}

inline Graph* Value::owningGraph() {
  return node()->owningGraph();
}

inline const Graph* Value::owningGraph() const {
  return node()->owningGraph();
}

/************* All nodes not required to be defined before Graph **************/
struct ProfileOp : public Node {
  static const Symbol Kind;
  ProfileOp(Graph* graph, std::function<void(std::vector<IValue>&)> callback)
      : Node(graph, ::c10::prim::profile), callback_(std::move(callback)) {}

  void cloneFrom(Node* other_) override;
  Node* allocNewInstance(Graph* g) override;

  const std::function<void(std::vector<IValue>&)>& getCallback() const {
    return callback_;
  }

  void setCallback(std::function<void(std::vector<IValue>&)> callback) {
    callback_ = std::move(callback);
  }

  bool hasSeenTensor() const {
    return has_seen_tensor_;
  }

  void setHasSeenTensor(bool has_seen_tensor) {
    has_seen_tensor_ = has_seen_tensor;
  }

 private:
  std::function<void(std::vector<IValue>&)> callback_;
  bool has_seen_tensor_ = false;
};

struct TORCH_API ProfileIValueOp : public Node {
  static const Symbol Kind;
  ProfileIValueOp(
      Graph* graph,
      std::function<void(std::vector<IValue>&)> callback)
      : Node(graph, ::c10::prim::profile_ivalue),
        callback_(std::move(callback)) {}

  void cloneFrom(Node* other_) override;
  Node* allocNewInstance(Graph* g) override;

  const std::function<void(std::vector<IValue>&)>& getCallback() const {
    return callback_;
  }

  void setCallback(std::function<void(std::vector<IValue>&)> callback) {
    callback_ = callback;
  }

 private:
  std::function<void(std::vector<IValue>&)> callback_;
};

// execute a Python function, used for Ops we can't optimize but that we want to
// optimize around
//
// Note: actual implementation (ConcretePythonOp) is defined in python_ir.cpp
// which is not included in libtorch.so. We still include some bits and pieces
// of PythonOp here to enable writing simple passes generically. In general,
// python-aware bits need to be moved to the descendant classes.
struct TORCH_API PythonOp : public Node {
  using Node::Node;

  virtual std::string name() const = 0;
  virtual void writeScalars(std::ostream& out) const = 0;
  void cloneFrom(Node* other_) override = 0;
  Node* allocNewInstance(Graph* g) override = 0;
  // recover the autograd.Function instance, if this PythonOp's function
  // was originally SomeFunction.apply
  // used in ONNX for discovering symbolics
  virtual std::optional<THPObjectPtr> autogradFunction() const = 0;

  virtual void lint_python() const = 0;
};

TORCH_API void LintGraph(const std::shared_ptr<Graph>& graph);

TORCH_API at::ArrayRef<Value*> createTupleUnpack(Value* v);

/** Insert graph \p CALLEE into graph \p G using \p INPUTS as input values.
 * The insertion happens at the current insertion point.
 * Optionally, one can also pass \p VALUE_MAP to get a map between \p CALLEE
 * values and their cloned copies in \p G.
 */
TORCH_API std::vector<Value*> insertGraph(
    Graph& g,
    Graph& callee,
    ArrayRef<Value*> inputs);
TORCH_API std::vector<Value*> insertGraph(
    Graph& g,
    Graph& callee,
    ArrayRef<Value*> inputs,
    std::unordered_map<Value*, Value*>& value_map);

/** If there is only one value in \p OUTPUTS and its kind is Tuple, insert a
 * tuple unpack node and return the resulting values.
 */
TORCH_API std::vector<Value*> unpackOutputs(const std::vector<Value*>& outputs);

template <typename T>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct FunctionSchemaMap {
  // Type aliasing
  using FuncSchemaMapType = typename std::pair<FunctionSchema, T>;
  using ValueType = std::vector<FuncSchemaMapType>;
  using MapType = std::unordered_map<Symbol, ValueType>;

  FunctionSchemaMap() = default;
  void insert(const FunctionSchema& schema, T val) {
    // Remove if exists before insert
    erase(schema);
    map[Symbol::fromQualString(schema.name())].emplace_back(
        std::make_pair(schema, val));
  }

  void erase(const FunctionSchema& schema) {
    auto it = map.find(Symbol::fromQualString(schema.name()));
    if (it == map.end()) {
      return;
    }
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first == schema) {
        it->second.erase(vit);
        break;
      }
    }
    if (it->second.size() == 0) {
      map.erase(Symbol::fromQualString(schema.name()));
    }
  }

  bool contains(const FunctionSchema& schema) const {
    const auto it = map.find(Symbol::fromQualString(schema.name()));
    if (it == map.end()) {
      return false;
    }
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first->schema() == schema) {
        return true;
      }
    }
    return false;
  }

  std::optional<T> find(const FunctionSchema& schema) const {
    const auto it = map.find(Symbol::fromQualString(schema.name()));
    if (it == map.end()) {
      return std::nullopt;
    }
    for (auto vit = it->second.begin(); vit != it->second.end(); ++vit) {
      if (vit->first == schema) {
        return vit->second;
      }
    }
    return std::nullopt;
  }

  // TODO: return iterator
  std::vector<FuncSchemaMapType> getAllKeysAndValues() const {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<FuncSchemaMapType> keys_values;
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

} // namespace jit
} // namespace habana_torch

CREATE_OSTREAM_FORMATTER(habana_torch::jit::Graph);
CREATE_OSTREAM_FORMATTER(habana_torch::jit::Node);
