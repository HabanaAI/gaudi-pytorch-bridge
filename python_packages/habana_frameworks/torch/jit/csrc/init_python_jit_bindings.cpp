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

// todo cleanup https://jira.habana-labs.com/browse/SW-199903
// probably should be moved to other place, optionally change file name
// to sth more appropriate

#include "init_python_jit_bindings.h"

#include <pybind11/stl.h>

#include <ATen/core/symbol.h>
#include <c10/core/MemoryFormat.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include "jit_fork/ir/ir.h"
#include "jit_fork/ir/irparser.h"
#include "jit_fork/ir/type_wrapper.h"
#include "jit_fork/passes/getitem_folding_pass.h"
#include "jit_fork/passes/rem_dup_const_pass.h"

#include <iostream>

namespace py = pybind11;

using Node = habana_torch::jit::Node;
using Graph = habana_torch::jit::Graph;
using Block = habana_torch::jit::Block;
using Value = habana_torch::jit::Value;

namespace habana_torch::jit {

// This is a variant of shared_ptr that "sees through" a wrapper.
// We use it to convert Value, Node, Block and node to "wrapped" Python
// values. When we destruct the C++ object, the wrapper's pointer will
// be set to 0 and any future dereferencing will throw. We need this
// because the Python objects may hang around after the C++ object
// has already been destroyed.
// This also needs the magic type_caster below, which is from the
// workaround offered in https://github.com/pybind/pybind11/issues/2751
template <typename T>
class unwrapping_shared_ptr {
  static_assert(
      std::is_same<T, Value>::value || std::is_same<T, Node>::value ||
          std::is_same<T, Block>::value,
      "unwrapping type only defined for Graph object types");

 private:
  std::shared_ptr<Wrap<T>> impl;

 public:
  unwrapping_shared_ptr() : impl({}) {}

  explicit unwrapping_shared_ptr(T* p) : impl(p->wrap()) {
    impl->clear_cb = &torch::jit::clear_registered_instances;
  }

  T* get() const {
    if (!impl->elem) {
      throw std::logic_error("has been invalidated");
    }
    return impl->elem;
  }

  // we need to disable the overloaded & for PyBind11 < 2.3 due.
  // see https://github.com/pybind/pybind11/pull/1435
#if (PYBIND11_VERSION_MAJOR > 2) || \
    ((PYBIND11_VERSION_MAJOR == 2) && (PYBIND11_VERSION_MINOR >= 3))
  T** operator&() {
    if (!impl->elem) {
      throw std::logic_error("has been invalidated");
    }
    return &(impl->elem);
  }
#endif
};
} // namespace habana_torch::jit

PYBIND11_DECLARE_HOLDER_TYPE(
    T,
    habana_torch::jit::unwrapping_shared_ptr<T>,
    true);

namespace pybind11 {
namespace detail {

#define CREATE_UNWRAPPING_CASTER(Class)                                                   \
  template <>                                                                             \
  struct type_caster<Class> : public type_caster_base<Class> {                            \
   public:                                                                                \
    using type = Class;                                                                   \
    using holder_type = habana_torch::jit::unwrapping_shared_ptr<Class>;                  \
                                                                                          \
    bool load(handle src, bool convert) {                                                 \
      return load_impl<type_caster<Class>>(src, convert);                                 \
    }                                                                                     \
                                                                                          \
    explicit operator type*() {                                                           \
      return static_cast<type*>(value);                                                   \
    }                                                                                     \
    explicit operator type&() {                                                           \
      return *static_cast<type*>(value);                                                  \
    }                                                                                     \
                                                                                          \
   protected:                                                                             \
    friend class type_caster_generic;                                                     \
                                                                                          \
    bool load_value(value_and_holder&& v_h) {                                             \
      if (v_h.holder_constructed()) {                                                     \
        value = v_h.template holder<holder_type>().get();                                 \
        return true;                                                                      \
      } else {                                                                            \
        throw cast_error(                                                                 \
            "Unable to cast from non-held to held instance (#Class& to Holder<#Class>)"); \
      }                                                                                   \
    }                                                                                     \
  }

// CREATE_UNWRAPPING_CASTER(Node);
CREATE_UNWRAPPING_CASTER(Value);
// CREATE_UNWRAPPING_CASTER(Block);

#undef CREATE_UNWRAPPING_CASTER

template <>
struct type_caster<habana_torch::jit::IValue> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(habana_torch::jit::IValue, _("IValue"));

  bool load(handle src, bool) {
    try {
      value = torch::jit::toTypeInferredIValue(src);
      return true;
    } catch (std::exception& e) {
      return false;
    }
  }

  static handle cast(
      habana_torch::jit::IValue src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return torch::jit::toPyObject(std::move(src)).release();
  }
};

template <>
struct type_caster<habana_torch::jit::Symbol> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(habana_torch::jit::Symbol, _("Symbol"));

  bool load(handle src, bool) {
    // TODO: Is there a way to py::cast that doesn't raise an exception on
    // failure?  Can we catch pybind11::cast_error here instead?
    std::string src_str;
    try {
      src_str = py::cast<std::string>(src);
    } catch (std::exception& e) {
      return false;
    }
    value = habana_torch::jit::Symbol::fromQualString(src_str);
    return true;
  }

  static handle cast(
      habana_torch::jit::Symbol src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(std::string(src.toQualString()), return_value_policy::copy)
        .release();
  }
};

template <>
struct type_caster<habana_torch::jit::AttributeKind> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(habana_torch::jit::AttributeKind, _("AttributeKind"));

  bool load(handle, bool) {
    return false;
  }

  static handle cast(
      habana_torch::jit::AttributeKind src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(
               std::string(habana_torch::jit::toString(src)),
               return_value_policy::copy)
        .release();
  }
};

// See https://github.com/pybind/pybind11/issues/637
using ListCasterBase = pybind11::detail::list_caster<
    std::vector<habana_torch::jit::Node*>,
    habana_torch::jit::Node*>;
template <>
struct type_caster<std::vector<habana_torch::jit::Node*>> : ListCasterBase {
  static handle cast(
      const std::vector<habana_torch::jit::Node*>& src,
      return_value_policy,
      handle parent) {
    return ListCasterBase::cast(src, return_value_policy::reference, parent);
  }
  static handle cast(
      const std::vector<habana_torch::jit::Node*>* src,
      return_value_policy pol,
      handle parent) {
    return cast(*src, pol, parent);
  }
};

} // namespace detail
} // namespace pybind11

namespace habana_torch {
namespace jit {

Node* findNode(c10::ArrayRef<Block*> blocks, Symbol kind, bool recurse = true) {
  for (Block* block : blocks) {
    for (Node* n : block->nodes()) {
      if (n->kind() == kind) {
        return n;
      }
      if (recurse) {
        auto node = findNode(n->blocks(), kind, recurse);
        if (node != nullptr) {
          return node;
        }
      }
    }
  }
  return nullptr;
}

Node* findNode(Block* block, Symbol kind, bool recurse = true) {
  std::vector<Block*> blocks = {block};
  return findNode(blocks, kind, recurse);
}

Node* addNodeToBlock(Block* block, Symbol kind, ArrayRef<Value*> inputs) {
  auto new_node = block->appendNode(block->owningGraph()->create(kind));
  for (auto input : inputs) {
    new_node->addInput(input);
  }
  return new_node;
}

void defineGraphClass(pybind11::module& m) {
#define GS(name) def(#name, &Graph ::name)

  py::class_<Graph, std::shared_ptr<Graph>>(m, "Graph", py::module_local())
      .def(py::init<>())
      .def(
          py::init([](const std::string& graph_string) {
            auto g = std::make_shared<Graph>();
            parseIR(graph_string, &*g, true /*parse_tensor_constants*/);
            return g;
          }),
          py::arg("graph_string"))
      .def(
          "__repr__",
          [&](Graph& g) { return g.toString(false /*print_source_info*/); })
      .def("str", &Graph::toString, py::arg("print_source_info") = false)
      .def(
          "inputs",
          [](Graph& g) {
            return py::make_iterator(g.inputs().begin(), g.inputs().end());
          },
          // We keep the graph alive while the iterator lives. Destroying
          // nodes might still be hazardous.
          py::keep_alive<0, 1>())
      .def(
          "outputs",
          [](Graph& g) {
            return py::make_iterator(g.outputs().begin(), g.outputs().end());
          },
          // We keep the graph alive while the iterator lives.
          // Destroying nodes might still be hazardous.
          py::keep_alive<0, 1>())
      .def(
          "nodes",
          [](Graph& g) {
            return py::make_iterator(g.nodes().begin(), g.nodes().end());
          },
          // We keep the graph alive while the iterator lives. Destroying
          // nodes might still be hazardous.
          py::keep_alive<0, 1>())
      .def(
          "findNode",
          [](Graph& g, const std::string& kind, bool recurse) {
            return findNode(g.block(), Symbol::fromQualString(kind), recurse);
          },
          "Find Node",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def(
          "findAllNodes",
          [](Graph& g, const std::string& kind, bool recurse) {
            return findAllNodes(g, Symbol::fromQualString(kind), recurse);
          },
          "Find all nodes",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def(
          "addInput",
          [](Graph& g, const std::string& name) { return g.addInput(name); },
          "Add input to graph with optional name seed",
          py::arg("name") = "")
      .def("copy", [](Graph& g) { return g.copy(); })
      .GS(eraseInput)
      .GS(eraseOutput)
      .GS(registerOutput)
      .def(
          "permuteInputs",
          [](Graph& g, const std::vector<size_t>& new_inputs) {
            g.block()->permuteInputs(new_inputs);
          })
      .def(
          "create",
          [](Graph& g, const char* str) {
            return g.create(Symbol::fromQualString(str));
          })
      .def(
          "create",
          [](Graph& g, const char* str, size_t noutputs) {
            return g.create(Symbol::fromQualString(str), noutputs);
          })
      .def(
          "create",
          [](Graph& g, const char* str, const std::vector<Value*>& inputs) {
            TORCH_CHECK_VALUE(
                std::all_of(
                    inputs.begin(),
                    inputs.end(),
                    [](Value* v) { return (v != nullptr); }),
                "cannot pass None in inputs");
            return g.create(Symbol::fromQualString(str), inputs);
          })
      .def(
          "create",
          [](Graph& g,
             const char* str,
             const std::vector<Value*>& inputs,
             size_t noutputs) {
            /*TORCH_CHECK_VALUE(
                std::all_of(
                    inputs.begin(),
                    inputs.end(),
                    [](Value* v) { return (v != nullptr); }),
                "cannot pass None in inputs");*/
            return g.create(Symbol::fromQualString(str), inputs, noutputs);
          })
      .def(
          "createList",
          [](Graph& g,
             const torch::jit::TypePtr& contained_type,
             const std::vector<Value*>& inputs) {
            /*TORCH_CHECK_VALUE(
                std::all_of(
                    inputs.begin(),
                    inputs.end(),
                    [](Value* v) { return (v != nullptr); }),
                "cannot pass None in inputs");*/
            return g.createList(contained_type, inputs);
          })
      .def(
          "createListUnpack",
          [](Graph& g, Value* list, size_t list_size) {
            return g.createListUnpack(list, list_size);
          })
      .def(
          "createTuple",
          [](Graph& g, const std::vector<Value*>& inputs) {
            return g.createTuple(inputs);
          })
      .def(
          "createTuple",
          [](Graph& g,
             const std::vector<Value*>& inputs,
             const TupleTypePtr& tuple_type) {
            return g.createTuple(inputs, tuple_type);
          })
      .def(
          "createTupleUnpack",
          [](Graph& g, Value* tuple) { return g.createTupleUnpack(tuple); })
      .def(
          "packValues",
          [](Graph& g, const std::vector<Value*>& inputs) {
            return g.packValues(inputs);
          })
      .def("param_node", [](Graph& g) { return g.block()->param_node(); })
      .def("return_node", [](Graph& g) { return g.block()->return_node(); })
      .def(
          "createClone",
          [](Graph& g, Node* n, py::object fn) {
            return g.createClone(
                n, [&](Value* e) { return fn(e).cast<Value*>(); });
          })
      .GS(appendNode)
      .GS(prependNode)
      .GS(insertPoint)
      .def("setInsertPoint", [](Graph& g, Node* n) { g.setInsertPoint(n); })
      .def("setInsertPoint", [](Graph& g, Block* n) { g.setInsertPoint(n); })
      .def(
          "insertGraph",
          [](Graph& g, Graph& callee, std::vector<Value*> inputs) {
            return insertGraph(g, callee, inputs);
          })
      .def(
          "insertGraph",
          [](Graph& g,
             Graph& callee,
             std::vector<Value*> inputs,
             std::unordered_map<Value*, Value*> value_map) {
            return insertGraph(g, callee, inputs, value_map);
          })
      .def(
          "insert",
          [](Graph& g, Symbol opname, std::vector<Value*> args) {
            std::vector<NamedValue> args_named;
            args_named.reserve(args.size());
            for (Value* v : args) {
              args_named.emplace_back(v);
            }
            return g.insert(opname, args_named);
          })
      .def(
          "insert",
          [](Graph& g,
             Symbol opname,
             std::vector<Value*> args,
             std::vector<NamedValue> kwargs) {
            std::vector<NamedValue> args_named;
            args_named.reserve(args.size());
            for (Value* v : args) {
              args_named.emplace_back(v);
            }
            return g.insert(opname, args_named, kwargs);
          })
      .def(
          "makeMultiOutputIntoTuple",
          [](Graph& g) {
            auto tup = g.createTuple(g.outputs());
            tup->insertBefore(g.return_node());
            for (int64_t i = g.outputs().size() - 1; i >= 0; i--) {
              g.eraseOutput(0);
            }
            g.registerOutput(tup->output());
          })
      .def(
          "insertConstant",
          [](Graph& g,
             const py::handle& handle,
             const torch::jit::TypePtr& jit_type) {
            IValue ival = torch::jit::toIValue(handle, jit_type);
            return g.insertConstant(ival);
          })
      .GS(lint)
      .def("block", [](Graph& g) { return g.block(); })
      .GS(insertNode);
#undef GS
}

void defineBlockClass(pybind11::module& m) {
  py::class_<Block, unwrapping_shared_ptr<Block>>(
      m, "Block", py::module_local())
      .def(
          "nodes",
          [](Block& b) {
            return py::make_iterator(b.nodes().begin(), b.nodes().end());
          })
      .def(
          "findNode",
          [](Block& b, const std::string& kind, bool recurse) {
            return findNode(&b, Symbol::fromQualString(kind), recurse);
          },
          "Find Node",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def(
          "findAllNodes",
          [](Block& b, const std::string& kind, bool recurse) {
            return findAllNodes(b, Symbol::fromQualString(kind), recurse);
          },
          "Find all nodes",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def(
          "inputs",
          [](Block& b) {
            return py::make_iterator(b.inputs().begin(), b.inputs().end());
          })
      .def(
          "outputs",
          [](Block& b) {
            return py::make_iterator(b.outputs().begin(), b.outputs().end());
          })
      .def("returnNode", [](Block& b) { return b.return_node(); })
      .def("paramNode", [](Block& b) { return b.param_node(); })
      .def("owningNode", [](Block& b) { return b.owningNode(); })
      .def(
          "addNode",
          [](Block& b, const char* str, const std::vector<Value*>& inputs) {
            return addNodeToBlock(&b, Symbol::fromQualString(str), inputs);
          })
      .def("addInputToBlock", [](Block& b) { return b.addInput(); })
      .def("registerOutput", [](Block& b, Value* value) {
        return b.registerOutput(value);
      });
}

void defineNodeClass(pybind11::module& m) {
#define NS(name) def(#name, &Node ::name)
  py::class_<Node, unwrapping_shared_ptr<Node>>(m, "Node", py::module_local())
      .def(
          "__repr__",
          [](Node& n) {
            std::stringstream ss;
            ss << n;
            return ss.str();
          })
      .def("setStackTrace", &Node::setStackTrace)
      .def("hasMultipleOutputs", [](Node& n) { return n.outputs().size() > 1; })
      .def("inputsSize", [](Node& n) { return n.inputs().size(); })
      .def("outputsSize", [](Node& n) { return n.outputs().size(); })
      .NS(kind)
      .def("prev", [](Node& n) { return n.prev(); })
      .def("matches", [](Node& n, const char* s) { return n.matches(s); })
      .def("owningBlock", [](Node& n) { return n.owningBlock(); })
      .def("inputsAt", [](Node& n, size_t i) { return n.inputs().at(i); })
      .def(
          "findNode",
          [](Node& n, const std::string& kind, bool recurse) {
            return findNode(n.blocks(), Symbol::fromQualString(kind), recurse);
          },
          "Find Node",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def(
          "findAllNodes",
          [](Node& n, const std::string& kind, bool recurse) {
            return findAllNodes(
                n.blocks(), Symbol::fromQualString(kind), recurse);
          },
          "Find all nodes",
          py::arg("kind"),
          py::arg("recurse") = true)
      .def("inputsSize", [](Node& n) { return n.inputs().size(); })
      .def("inputsAt", [](Node& n, size_t i) { return n.inputs().at(i); })
      .def("input", [](Node& n) { return n.input(); })
      .def(
          "inputs",
          [](Node& n) {
            return py::make_iterator(n.inputs().begin(), n.inputs().end());
          })
      .def("outputsSize", [](Node& n) { return n.outputs().size(); })
      .def("outputsAt", [](Node& n, size_t i) { return n.outputs().at(i); })
      .def("output", [](Node& n) { return n.output(); })
      .def(
          "outputs",
          [](Node& n) {
            return py::make_iterator(n.outputs().begin(), n.outputs().end());
          })
      .def(
          "schema",
          [](Node& n) {
            std::stringstream ss;
            if (n.maybeSchema()) {
              ss << n.schema();
            } else {
              ss << "(no schema)";
            }
            return ss.str();
          })
      .def(
          "getModuleHierarchy",
          [](Node& n) { return utils::getNodesModuleHierarchy(n); })
      .def(
          "namedInput",
          [](Node& n, const std::string& unqualName) {
            return n.namedInput(unqualName);
          })
      .NS(addInput)
      .NS(copyMetadata)
      .NS(replaceInput)
      .NS(replaceInputWith)
      .NS(replaceAllUsesWith)
      .NS(insertBefore)
      .NS(insertAfter)
      .NS(isBefore)
      .NS(isAfter)
      .NS(moveAfter)
      .NS(moveBefore)
      .NS(removeInput)
      .NS(removeAllInputs)
      .NS(destroy)
      .NS(hasUses)
      .NS(eraseOutput)
      .NS(addOutput)
      .NS(scopeName)
      .NS(isNondeterministic)
      .def(
          "blocks",
          [](Node& n) {
            return py::make_iterator(n.blocks().begin(), n.blocks().end());
          })
      .NS(addBlock)
      .NS(mustBeNone)

#define AS(name) def(#name, &Node::name)
      // methods from Attributes
      .AS(copyAttributes)
      .AS(hasAttributes)
#undef AS
#define AS(name) def(#name, &Node::name##S)
      // The default method names take Symbol, but the string conversion for
      // Symbol you to qualify with attr::. This is not very user friendly
      // for attributes, so expose the string variants instead.
      .AS(hasAttribute)
      .AS(kindOf)
      .AS(removeAttribute)
      .AS(attributeNames)
#undef AS
#define CREATE_ACCESSOR(Kind, method)                                       \
  def(#method "_", [](Node& n, const char* name, Kind##Attr::ValueType v) { \
    return n.method##_(Symbol::attr(name), std::move(v));                   \
  }).def(#method, [](Node& n, const char* name) {                           \
    return n.method(Symbol::attr(name));                                    \
  })
      .CREATE_ACCESSOR(Float, f)
      .CREATE_ACCESSOR(Floats, fs)
      .CREATE_ACCESSOR(Complex, c)
      .CREATE_ACCESSOR(String, s)
      .CREATE_ACCESSOR(Strings, ss)
      .CREATE_ACCESSOR(Int, i)
      .CREATE_ACCESSOR(Ints, is)
      .CREATE_ACCESSOR(Graph, g)
      .CREATE_ACCESSOR(Graphs, gs)
      .CREATE_ACCESSOR(IValue, ival)
#undef CREATE_ACCESSOR
      // Tensor (t_) -- manually written to unwrap the variable into a tensor.
      .def(
          "t_",
          [](Node& n, const char* name, const torch::autograd::Variable& v) {
            AT_ASSERT(!v.requires_grad());
            return n.t_(Symbol::attr(name), v);
          })
      .def(
          "t",
          [](Node& n, const char* name) { return n.t(Symbol::attr(name)); })
      // Tensors (ts_) -- manually written to unwrap variables into tensors.
      .def(
          "ts_",
          [](Node& n,
             const char* name,
             const std::vector<torch::autograd::Variable>& vs) {
            std::vector<at::Tensor> tensors;
            tensors.reserve(vs.size());
            for (auto& variable : vs) {
              AT_ASSERT(!variable.requires_grad());
              tensors.push_back(variable);
            }
            return n.ts_(Symbol::attr(name), std::move(tensors));
          })
      .def(
          "ts",
          [](Node& n, const char* name) {
            auto tensors = n.ts(Symbol::attr(name));
            std::vector<torch::autograd::Variable> variables;
            variables.reserve(tensors.size());
            for (auto& tensor : tensors) {
              variables.emplace_back(std::move(tensor));
            }
            return variables;
          })
      .def(
          "z_",
          [](Node& n, const char* name, const at::Tensor& v) {
            return n.t_(
                Symbol::attr(name),
                torch::autograd::Variable(v.view(std::vector<int64_t>{}))
                    .set_requires_grad(false));
          })
      .def(
          "z",
          [](Node& n, const char* name) { return n.t(Symbol::attr(name)); })
      .def(
          "ty_",
          [](Node& n, const char* name, const TypeWrapper& type) {
            return n.ty_(Symbol::attr(name), type);
          })
      .def(
          "ty",
          [](Node& n, const char* name) { return n.ty(Symbol::attr(name)); })
      .def(
          "tys_",
          [](Node& n, const char* name, const std::vector<TypeWrapper>& types) {
            return n.tys_(Symbol::attr(name), types);
          })
      .def(
          "tys",
          [](Node& n, const char* name) { return n.tys(Symbol::attr(name)); })
      .def(
          "zs_",
          [](Node& n, const char* name, TensorsAttr::ValueType v) {
            for (auto& i : v) {
              i = torch::autograd::Variable(i.view(std::vector<int64_t>{}))
                      .set_requires_grad(false);
            }
            return n.ts_(Symbol::attr(name), std::move(v));
          })
      .def("zs", [](Node& n, const char* name) {
        return n.ts(Symbol::attr(name));
      });
#undef NS
}

void defineValueClass(pybind11::module& m) {
#define VS(name) def(#name, &Value ::name)
  py::class_<Value, unwrapping_shared_ptr<Value>>(
      m, "Value", py::module_local())
      .def(
          "__repr__",
          [](Value& n) {
            std::stringstream ss;
            ss << n.debugName() << " defined in (" << *n.node() << ")";
            return ss.str();
          })
      .VS(type)
      .def("setType", py::overload_cast<TypePtr>(&Value::setType))
      .def("setType", py::overload_cast<const TypeWrapper&>(&Value::setType))
      .def(
          "inferTypeFrom",
          py::overload_cast<const at::Tensor&>(&Value::inferTypeFrom))
      .def(
          "inferTypeFrom",
          py::overload_cast<const c10::intrusive_ptr<c10::ivalue::Object>&>(
              &Value::inferTypeFrom))
      .def(
          "setDebugName",
          &Value::setDebugName,
          py::arg("name"),
          py::arg("allow_numbers") = false)
      // skip owningGraph because it returns a raw pointer to a otherwise
      // std::shared_ptr stored graph object, and would cause a double free
      .VS(unique)
      .VS(debugName)
      .VS(offset)
      .VS(uses)
      .VS(replaceAllUsesWith)
      .VS(replaceAllUsesAfterNodeWith)
      .def("node", [](Value& v) { return v.node(); })
      .def(
          "setTypeAs",
          [](Value* node, Value* other) {
            node->setType(other->type());
            return node;
          })
      .VS(copyMetadata)
      .VS(isCompleteTensor)
      .VS(requires_grad)
      .def(
          "requiresGrad",
          [](Value& n) {
            return n.type()->expectRef<c10::TensorType>().requiresGrad();
          })
      .def("toIValue", [](Value& n) { return toIValue(&n); })
      .def("type", [](Value& v) { return v.type(); });
#undef VS
}

void defineNamedValueClass(pybind11::module& m) {
  py::class_<NamedValue>(m, "NamedValue", py::module_local())
      .def(
          py::init([](std::string name, unwrapping_shared_ptr<Value> value) {
            return NamedValue(name, value.get());
          }),
          py::arg("name"),
          py::arg("value"))
      .def("name", &NamedValue::name)
      .def("value", &NamedValue::value);
}

void defineRealTypeClasses(pybind11::module& m) {
  using namespace c10;

  m.def(
      "check_if_types_are_matching",
      [](const torch::jit::TypePtr& lhs, const torch::jit::TypePtr& rhs) {
        return matchTypes(lhs, rhs);
      });

  py::class_<TypeWrapper>(m, "TypeWrapper", py::module_local())
      .def(py::init([](TypePtr type) { return TypeWrapper(type); }))
      .def(py::init([](TypePtr type, const std::string& symbol_or_expr) {
        return TypeWrapper(type, symbol_or_expr);
      }))
      .def(py::init([](const at::Tensor& tensor,
                       const py::list& py_shape,
                       const py::list& py_strides) {
        auto process_list = [](const py::list& py_list) {
          std::vector<DimVariants> cpp_list;
          cpp_list.reserve(py_list.size());
          for (const auto& item : py_list) {
            if (py::isinstance<py::str>(item)) {
              const std::string str_value = item.cast<std::string>();
              cpp_list.push_back(str_value);
            } else if (py::isinstance<py::int_>(item)) {
              const int64_t int_value = item.cast<int64_t>();
              cpp_list.push_back(int_value);
            } else {
              TORCH_CHECK(
                  0, "Shape or strides contain incorrect dimension info.");
            }
          }
          return cpp_list;
        };

        const SymbolicShape symbolic_shape = process_list(py_shape);
        const SymbolicStrides symbolic_strides = process_list(py_strides);

        return TypeWrapper::createTensorTypeWrapper(
            tensor.scalar_type(),
            symbolic_shape,
            symbolic_strides,
            tensor.device(),
            tensor.requires_grad());
      }));

  py::class_<NoneType, Type, NoneTypePtr>(m, "NoneType", py::module_local())
      .def_static("get", &NoneType::get);
  py::class_<AnyType, Type, AnyTypePtr>(m, "AnyType", py::module_local())
      .def_static("get", &AnyType::get);

  py::class_<BoolType, Type, BoolTypePtr>(m, "BoolType", py::module_local())
      .def_static("get", &BoolType::get);
  py::class_<NumberType, Type, NumberTypePtr>(
      m, "NumberType", py::module_local())
      .def_static("get", &NumberType::get);
  py::class_<IntType, Type, IntTypePtr>(m, "IntType", py::module_local())
      .def_static("get", &IntType::get);
  py::class_<FloatType, Type, FloatTypePtr>(m, "FloatType", py::module_local())
      .def_static("get", &FloatType::get);
  py::class_<ComplexType, Type, ComplexTypePtr>(
      m, "ComplexType", py::module_local())
      .def_static("get", &ComplexType::get);

  py::class_<SymBoolType, Type, SymBoolTypePtr>(
      m, "SymBoolType", py::module_local())
      .def_static("get", &SymBoolType::get);
  py::class_<SymIntType, Type, SymIntTypePtr>(
      m, "SymIntType", py::module_local())
      .def_static("get", &SymIntType::get);
  py::class_<SymFloatType, Type, SymFloatTypePtr>(
      m, "SymFloatType", py::module_local())
      .def_static("get", &SymFloatType::get);

  py::class_<StringType, Type, StringTypePtr>(
      m, "StringType", py::module_local())
      .def_static("get", &StringType::get);
  py::class_<LayoutType, Type, LayoutTypePtr>(
      m, "LayoutType", py::module_local())
      .def_static("get", &LayoutType::get);
  py::class_<MemoryFormatType, Type, MemoryFormatTypePtr>(
      m, "MemoryFormatType", py::module_local())
      .def_static("get", &MemoryFormatType::get);
  py::class_<DeviceObjType, Type, DeviceObjTypePtr>(
      m, "DeviceObjType", py::module_local())
      .def_static("get", &DeviceObjType::get);
  py::class_<StreamObjType, Type, StreamObjTypePtr>(
      m, "StreamObjType", py::module_local())
      .def_static("get", &StreamObjType::get);
  py::class_<TensorType, Type, TensorTypePtr>(
      m, "TensorType", py::module_local())
      .def_static("get", &TensorType::get)
      .def_static("getInferred", &TensorType::getInferred)
      .def_static("create_from_tensor", [](const at::Tensor& t) {
        return TensorType::create(t);
      });

  py::class_<OptionalType, Type, OptionalTypePtr>(
      m, "OptionalType", py::module_local())
      .def(py::init(
          [](TypePtr a) { return OptionalType::create(std::move(a)); }))
      .def_static("ofTensor", &OptionalType::ofTensor)
      .def("getElementType", &OptionalType::getElementType);

  py::class_<ListType, Type, ListTypePtr>(m, "ListType", py::module_local())
      .def(py::init([](TypePtr a) { return ListType::create(a); }))
      .def_static("ofInts", &ListType::ofInts)
      .def_static("ofTensors", &ListType::ofTensors)
      .def_static("ofFloats", &ListType::ofFloats)
      .def_static("ofComplexDoubles", &ListType::ofComplexDoubles)
      .def_static("ofBools", &ListType::ofBools)
      .def_static("ofStrings", &ListType::ofStrings)
      .def("getElementType", &ListType::getElementType);
  py::class_<TupleType, Type, TupleTypePtr>(m, "TupleType", py::module_local())
      .def(py::init([](std::vector<TypePtr> types) {
        return TupleType::create(std::move(types));
      }))
      .def(py::init([](const std::string& name,
                       const std::vector<std::string>& fields,
                       const std::vector<TypePtr>& types) {
        return TupleType::createNamed(name, fields, types);
      }))
      .def("elements", [](TupleType& self) {
        std::vector<TypePtr> types;
        for (const auto& type : self.elements()) {
          types.push_back(type);
        }
        return types;
      });
  py::class_<DictType, Type, DictTypePtr>(m, "DictType", py::module_local())
      .def(py::init([](TypePtr key, TypePtr value) {
        return DictType::create(std::move(key), std::move(value));
      }))
      .def("getKeyType", &DictType::getKeyType)
      .def("getValueType", &DictType::getValueType);
}

void defineJitPasses(pybind11::module& m) {
  m.def(
      "remove_duplicate_const_pass",
      &habana_torch::jit::RemoveDuplicateConstPass,
      py::arg("graph"));
  m.def(
      "getitem_folding_pass",
      &habana_torch::jit::GetItemFoldingPass,
      py::arg("graph"));
}

void InitBindings(py::module& m) {
  auto m_jit = m.def_submodule("jit");

  defineGraphClass(m_jit);
  defineBlockClass(m_jit);
  defineNodeClass(m_jit);
  defineValueClass(m_jit);
  defineNamedValueClass(m_jit);
  defineRealTypeClasses(m_jit);
  defineJitPasses(m_jit);
}

} // namespace jit
} // namespace habana_torch
