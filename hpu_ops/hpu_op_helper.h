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
using sizes_vec = std::vector<std::vector<int64_t>>;

inline at::Tensor& stack_tensor(at::Stack& stack, int index) {
  return stack.at(index).toTensor();
}

inline at::Tensor stack_tensor(const at::Stack& stack, int index) {
  return stack.at(index).toTensor();
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

class HabanaOperatorHelper : public HabanaOperator {
 public:
  HabanaOperatorHelper(
      int device_id,
      const std::string& guid,
      c10::ScalarType scalar_type,
      int out_id,
      int inplace_id,
      int scalar_id,
      bool is_outfn)
      : HabanaOperator(
            guid + habana_helpers::name_suffix_from_type(scalar_type)),
        m_scalar_type{scalar_type},
        m_out_id{out_id},
        m_inplace_id{inplace_id},
        m_scalar_id{scalar_id},
        m_is_outfn{is_outfn} {
    CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

 protected:
  const c10::ScalarType& ScalarType() const {
    return m_scalar_type;
  }

  const std::unordered_map<int, at::Scalar>& ScalarInputs() const {
    return m_scalar_inputs;
  }

  int ScalarId() const {
    return m_scalar_id;
  }

  bool IsOutFn() const {
    return m_is_outfn;
  }

  void set_layouts(
      std::vector<LayoutFormat> in_layouts,
      std::vector<LayoutFormat> out_layouts) {
    kernel_meta_data_.input_layout = std::move(in_layouts);
    kernel_meta_data_.output_layout = std::move(out_layouts);
  }

 private:
  virtual void CustomHandler(synapse_helpers::graph&, at::Stack&) {}

  virtual std::shared_ptr<void> FillParams(const at::Stack&, size_t& size) {
    size = 0;
    return nullptr;
  }

  virtual sizes_vec ComputeOutputShapes(const at::Stack&) {
    return {};
  }

  void HandleScalarToTensor(
      synapse_helpers::graph& graph,
      const at::Stack& stack);
  void HandleFn(
      synapse_helpers::graph& graph,
      const at::Stack& stack,
      const std::vector<bool>& is_output_persistent_list);
  void HandleInplaceFn(synapse_helpers::graph& graph, const at::Stack& stack);
  void HandleOutFn(synapse_helpers::graph& graph, const at::Stack& stack);

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

  // Compound node helpers
  struct _node_output_attr {
    at::IntArrayRef sizes{};
    at::ScalarType dtype{at::kFloat};
    bool persistent{false};
    bool final_node{false};
  };

 protected:
  std::vector<synapse_helpers::tensor> BuildOp(
      synapse_helpers::graph& graph,
      const std::string& guid,
      std::vector<synTensor> node_inputs,
      const std::vector<_node_output_attr>& node_output_attrs,
      void* params = nullptr,
      size_t param_size = 0);

  synTensor& syn_in(int index) {
    return p_context_->syn_inputs_.at(index).ref().get();
  }

  synapse_helpers::tensor& syn_out(int index) {
    return p_context_->syn_outputs_.at(index);
  }

  virtual void AddNode(
      synapse_helpers::graph&,
      at::Stack&,
      const std::vector<bool>&);

 private:
  const c10::ScalarType m_scalar_type;
  const int m_out_id;
  const int m_inplace_id;
  const int m_scalar_id;
  const bool m_is_outfn;

  std::unordered_map<int, at::Scalar> m_scalar_inputs;

 public:
  static std::shared_ptr<void> FillClampMaxParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillClampMinParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillClampParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillCumsumParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillGridSamplerParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillTriuParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillTrilParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillHardSigmoidParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillMseLossParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillEluBackwardParams(const at::Stack&, size_t&);

  static sizes_vec AddCOpsOutputShape(const at::Stack&, bool = false);
  static sizes_vec BinaryOutputShape(const at::Stack&, bool = false);
  static sizes_vec MseLossBwdOutputShape(const at::Stack&, bool = false);
  static sizes_vec MseLossOutputShape(const at::Stack&, bool = false);
  static sizes_vec PowOutputShape(const at::Stack&, bool = false);
  static sizes_vec GridSampler2dOutputShape(const at::Stack&, bool = false);
};

#define PARAMS_STUB(structname) \
  size = sizeof(structname);    \
  auto params = std::make_shared<structname>()

#define HPU_CUSTOM_HABANA_OP(op)            \
  struct op : HabanaOperatorHelper {        \
    op(int device_id,                       \
       const std::string& guid,             \
       c10::ScalarType scalar_type,         \
       int out_id,                          \
       int inplace_id,                      \
       int scalar_id,                       \
       bool is_outfn)                       \
        : HabanaOperatorHelper(             \
              device_id,                    \
              guid,                         \
              scalar_type,                  \
              out_id,                       \
              inplace_id,                   \
              scalar_id,                    \
              is_outfn){};                  \
    void AddNode(                           \
        synapse_helpers::graph&,            \
        at::Stack&,                         \
        const std::vector<bool>&) override; \
  };

} // namespace habana

#define HPU_FRONTEND_OP(op)                                                    \
  template <typename T>                                                        \
  struct op : habana_lazy::LazyOp<T> {                                         \
    op(const std::string& qualstring,                                          \
       const std::vector<at::IValue>& inputs,                                  \
       const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn = \
           {});                                                                \
    T get_result_overrideable() override;                                      \
  };

#define HPU_SUPPORTED_DTYPES(fn, supported_dtypes)                       \
  const static std::unordered_set<c10::ScalarType> fn##_supported_dtypes \
      supported_dtypes;

#define FALLBACK_IF_UNSUPPORTED_DTYPE(tensor, fn, args...)       \
  if (ABSL_PREDICT_FALSE(                                        \
          tensor.defined() &&                                    \
          !fn##_supported_dtypes.count(tensor.scalar_type()))) { \
    return AtenHpuTypeDefault::fn(args);                         \
  }
