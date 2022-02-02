/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <utility>
#include "habana_kernels/habana_operator.h"
#include "habana_kernels/kernel_utils.h"
namespace habana {

class SupportedDtypes {
 public:
  SupportedDtypes(std::unordered_set<c10::ScalarType> dtypes)
      : m_dtypes(std::move(dtypes)) {}
  bool count(c10::ScalarType type) const;
  bool count(const at::Tensor& tensor) const;
  bool count(const c10::optional<at::Tensor>& tensor) const;

 private:
  std::unordered_set<c10::ScalarType> m_dtypes;
};

using sizes_vec = std::vector<std::vector<int64_t>>;

inline at::Tensor& stack_tensor(at::Stack& stack, int index) {
  return stack.at(index).toTensor();
}

inline at::Tensor stack_tensor(const at::Stack& stack, int index) {
  return stack.at(index).toTensor();
}

inline std::string& update_guid_dtype(
    std::string& guid,
    const std::string& dtype_str) {
  guid = guid.substr(0, guid.find_last_of('_') + 1).append(dtype_str);
  return guid;
}

inline std::string& update_guid_dtype(
    std::string& guid,
    c10::ScalarType dtype) {
  return update_guid_dtype(guid, habana_helpers::name_suffix_from_type(dtype));
}

inline int get_dim_in_tpc_order(int64_t dim_, int64_t max_dims) {
  auto dim = at::maybe_wrap_dim(dim_, max_dims, /*wrap_scalar=*/true);
  return static_cast<int>(max_dims - dim - 1);
}

std::vector<at::Tensor> GetMetaTensorList(
    const std::vector<at::Tensor>& tensors);
std::vector<c10::optional<at::Tensor>> GetMetaOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors);

template <typename T>
T& get(fint_t&);

template <>
inline int& get<int>(fint_t& u) {
  return u.i;
}
template <>
inline float& get<float>(fint_t& u) {
  return u.f;
}

struct NodeAttr {
  struct NodeOutputAttr {
    at::IntArrayRef sizes{};
    at::ScalarType dtype{at::kFloat};
    c10::optional<int> final_result_index{c10::nullopt};
    synTensorType tensor_type{DATA_TENSOR};
  };

  std::string guid;
  std::vector<synTensor> inputs;
  std::vector<NodeOutputAttr> output_attrs;
  void* params = nullptr;
  size_t param_size = 0;
};

class OpBackend : public HabanaOperator {
 public:
  OpBackend(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalar_type,
      std::vector<int> res_ids,
      std::vector<int> inplace_ids,
      std::vector<int> scalar_ids,
      bool is_outfn);

 public:
  const c10::ScalarType& ScalarType() const {
    return m_scalar_type;
  }

 protected:
  c10::ScalarType ComputePromotedScalarType(
      const at::Stack& stack,
      bool update);

  const std::unordered_map<int, at::Scalar>& ScalarInputs() const {
    return m_scalar_inputs;
  }

  std::vector<int> ScalarId() const {
    return m_scalar_ids;
  }

  bool IsOutputAvailable() const {
    return m_is_outfn or m_inplace_ids.size();
  }

  bool IsOutputPersistent(int i) const {
    // Reuse from HabanaOperator::OutputMetaData when available
    return m_persistence_list[i];
  }

  void SetLayouts(
      std::vector<LayoutFormat> in_layouts,
      std::vector<LayoutFormat> out_layouts) {
    kernel_meta_data_.input_layout = std::move(in_layouts);
    kernel_meta_data_.output_layout = std::move(out_layouts);
  }

  void SetNumOutTensors(int n) {
    m_num_out_tensors = n;
  }

  void EnableTypePromotion() {
    m_promote_type = true;
  }

  void PromoteIntToFloat() {
    m_promote_int_to_float = true;
  }

  void SetFillParams(
      std::function<std::shared_ptr<void>(const at::Stack&, size_t&)> fn) {
    m_fill_params = std::move(fn);
  }

  std::shared_ptr<void> FillParams(const at::Stack& stack, size_t& size) {
    return m_fill_params ? m_fill_params(stack, size) : nullptr;
  }

  void SetComputeOutputShapes(
      std::function<sizes_vec(const at::Stack&, bool)> fn) {
    m_compute_output_shapes = std::move(fn);
  }

  sizes_vec ComputeOutputShapes(
      const at::Stack& stack,
      bool is_lowering = false) const {
    if (m_compute_output_shapes) {
      return m_compute_output_shapes(stack, is_lowering);
    }
    return {};
  }

  virtual void CustomHandler(synapse_helpers::graph&, at::Stack&) {}

 private:
  void HandleScalarToTensor(
      synapse_helpers::graph& graph,
      const at::Stack& stack);
  void HandleFn(synapse_helpers::graph& graph, const at::Stack& stack);
  void HandleInplaceFn(synapse_helpers::graph& graph, const at::Stack& stack);
  void HandleOutFn(synapse_helpers::graph& graph, const at::Stack& stack);
  void HandleTypePromotion(
      synapse_helpers::graph& graph,
      const at::Stack& stack);
  void HandleIntToFloatPromotion(
      synapse_helpers::graph& graph,
      const at::Stack& stack);

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      at::Stack& stack,
      bool is_output_persistent) override {
    std::vector<bool> is_output_persistent_list{is_output_persistent};
    AllocateAndAddSynapseNode(graph, stack, is_output_persistent_list);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      at::Stack& stack,
      std::vector<bool> is_output_persistent) override;

 protected:
  std::vector<synapse_helpers::tensor> BuildOp(
      synapse_helpers::graph& graph,
      const std::string& guid,
      std::vector<synTensor> node_inputs,
      const std::vector<NodeAttr::NodeOutputAttr>& node_output_attr,
      void* params = nullptr,
      size_t param_size = 0);

  synTensor& syn_in(int index) {
    return p_context_->syn_inputs_.at(index).ref().get();
  }

  synapse_helpers::tensor& syn_out(int index) {
    return p_context_->syn_outputs_.at(index);
  }

  synapse_helpers::tensor CastHelper(
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      const at::ScalarType& from,
      const at::ScalarType& to,
      c10::optional<int> final_result_index = c10::nullopt);

  synapse_helpers::tensor ConstantHelper(
      synapse_helpers::graph& graph,
      const at::Scalar& val,
      c10::optional<at::ScalarType> force_type = c10::nullopt,
      const at::IntArrayRef constant_outshape = 1,
      c10::optional<int> final_result_index = c10::nullopt);

  synapse_helpers::tensor ReshapeHelper(
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt);

  virtual void AddNode(synapse_helpers::graph&, const at::Stack&);

 public:
  static std::vector<synapse_helpers::tensor> BuildNode(
      OpBackend* op,
      synapse_helpers::graph& graph,
      NodeAttr node_attr);

  static synapse_helpers::tensor BuildCast(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      const at::IntArrayRef sizes,
      const at::ScalarType& from,
      const at::ScalarType& to,
      c10::optional<int> final_result_index = c10::nullopt);

  static synapse_helpers::tensor BuildConstant(
      OpBackend* op,
      synapse_helpers::graph& graph,
      const at::Scalar& val,
      c10::optional<at::ScalarType> force_type = c10::nullopt,
      const at::IntArrayRef constant_outshape = 1,
      c10::optional<int> final_result_index = c10::nullopt);

  static synapse_helpers::tensor BuildReshape(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt);

 private:
  const std::vector<int> m_res_ids;
  const std::vector<int> m_inplace_ids;
  const std::vector<int> m_scalar_ids;
  const bool m_is_outfn;

  c10::ScalarType m_scalar_type;
  bool m_promote_type = false;
  bool m_promote_int_to_float = false;
  int m_num_out_tensors = 1;

  std::unordered_map<int, at::Scalar> m_scalar_inputs;
  std::function<std::shared_ptr<void>(const at::Stack&, size_t&)> m_fill_params;
  std::function<sizes_vec(const at::Stack&, bool)> m_compute_output_shapes;
  std::vector<bool>
      m_persistence_list; // Reuse from HabanaOperator::OutputMetaData when
                          // available
};

#define PARAMS_STUB(structname) \
  size = sizeof(structname);    \
  auto params = std::make_shared<structname>()

// Use when you want to define your own size and param var names
#define PARAMS_STUB_VARS(structname, params_size, params) \
  const size_t& params_size = sizeof(structname);         \
  auto params = std::make_shared<structname>()

#define HPU_OP_BACKEND(op)                                            \
  struct op : OpBackend {                                             \
    op(int device_id,                                                 \
       const std::string& guid,                                       \
       c10::ScalarType scalar_type,                                   \
       const std::vector<int>& res_ids,                               \
       const std::vector<int>& inplace_ids,                           \
       const std::vector<int>& scalar_ids,                            \
       bool is_outfn)                                                 \
        : OpBackend(                                                  \
              device_id,                                              \
              guid,                                                   \
              scalar_type,                                            \
              res_ids,                                                \
              inplace_ids,                                            \
              scalar_ids,                                             \
              is_outfn){};                                            \
    void AddNode(synapse_helpers::graph&, const at::Stack&) override; \
  };

} // namespace habana

#define HPU_OP_FRONTEND(op)                                                    \
  template <typename T>                                                        \
  struct op : habana_lazy::LazyOp<T> {                                         \
    op(const std::string& qualstring,                                          \
       const std::vector<at::IValue>& inputs,                                  \
       const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn = \
           {});                                                                \
    T get_result_overrideable() override;                                      \
  };

#define FILL_PARAMS_DECL(fn) \
  std::shared_ptr<void> fn(const at::Stack&, size_t&);

#define OUTSHAPE_DECL(fn) sizes_vec fn(const at::Stack&, bool = false);

#define HPU_SUPPORTED_DTYPES(fn, supported_dtypes) \
  const static SupportedDtypes fn##_supported_dtypes supported_dtypes;

#define FALLBACK_IF_UNSUPPORTED_DTYPE(input, fn, args...)        \
  if (ABSL_PREDICT_FALSE(!fn##_supported_dtypes.count(input))) { \
    return AtenHpuTypeDefault::fn(args);                         \
  }

#define FALLBACK_IF_UNSUPPORTED_DTYPE_PER_TENSOR(tensor, fn, args...)    \
  if (ABSL_PREDICT_FALSE(                                                \
          tensor.defined() &&                                            \
          !fn##tensor##_supported_dtypes.count(tensor.scalar_type()))) { \
    return AtenHpuTypeDefault::fn(args);                                 \
  }

#define FALLBACK_IF_UNSUPPORTED_INPUTS(check_fn, op, args...) \
  if (ABSL_PREDICT_FALSE(!check_fn(args))) {                  \
    return AtenHpuTypeDefault::op(args);                      \
  }

#define FALLBACK_CHECK(check_fn, signature...)          \
  extern const std::function<bool(signature)> check_fn; \
  const std::function<bool(signature)> check_fn = [](signature)
