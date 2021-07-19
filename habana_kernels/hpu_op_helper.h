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
#include "habana_kernels/habana_operator.h"
#include "habana_kernels/kernel_utils.h"
namespace habana {
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

 private:
  virtual void CustomHandler(synapse_helpers::graph&, at::Stack&) {}

  virtual std::shared_ptr<void> FillParams(const at::Stack&, size_t& size) {
    size = 0;
    return nullptr;
  }

  void HandleScalarToTensor(
      synapse_helpers::graph& graph,
      const at::Stack& stack) {
    if (m_scalar_id < 0) {
      return;
    }

    // Get rid of this kludge
    const auto& const_op =
        make_operator<ConstantOperator>(p_context_->device_id_, m_scalar_type);
    const auto& t = at::detail::make_tensor<c10::TensorImpl>(
        c10::DispatchKeySet{
            at::DispatchKey::HABANATensorId, at::DispatchKey::AutogradHABANA},
        c10::scalarTypeToTypeMeta(m_scalar_type),
        c10::Device(c10::kHABANA, 0));
    t.unsafeGetTensorImpl()->set_sizes_contiguous(1);
    at::Stack s = {t, stack.at(m_scalar_id)};
    const_op->AllocateAndAddSynapseNode(graph, s, false);
    HABANA_ASSERT(
        m_scalar_id <= static_cast<int>(p_context_->syn_inputs_.size()));
    p_context_->syn_inputs_.emplace(
        p_context_->syn_inputs_.cbegin() + m_scalar_id,
        std::move(const_op->GetSynOutputs()[0]));
  }

  void HandleFn(
      synapse_helpers::graph& graph,
      const at::Stack& stack,
      bool is_output_persistent) {
    if (m_out_id < 0) {
      return;
    }
    const auto& output = habana_helpers::createPTTensor(
        stack.at(m_out_id).toTensor(), is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  }

  void HandleOutFn(const at::Stack& stack) {
    if (!m_is_outfn) {
      return;
    }

    p_context_->pt_outputs_.emplace_back(stack.back().toTensor());
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_.back()));
    p_context_->syn_inputs_.pop_back();
  }

  void HandleInplaceFn(const at::Stack& stack) {
    if (m_inplace_id < 0) {
      return;
    }
    // Index can vary in syn_inputs_ and in stack
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[m_inplace_id]));
    p_context_->pt_outputs_.emplace_back(stack[m_inplace_id].toTensor());
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      at::Stack& stack,
      bool is_output_persistent) override {
    CustomHandler(graph, stack);
    HandleFn(graph, stack, is_output_persistent);
    HandleInplaceFn(stack);
    HandleOutFn(stack);
    HandleScalarToTensor(graph, stack);

    size_t size = 0;
    const auto& params = FillParams(stack, size);
    AddNodeToSynapseGraph(graph, params.get(), size);
  }

 private:
  const c10::ScalarType m_scalar_type;
  const int m_out_id;
  const int m_inplace_id;
  const int m_scalar_id;
  const bool m_is_outfn;

 public:
  static std::shared_ptr<void> FillClampParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillClampMinParams(const at::Stack&, size_t&);
  static std::shared_ptr<void> FillClampMaxParams(const at::Stack&, size_t&);
};

std::vector<at::Tensor> GetMetaTensorList(
    const std::vector<at::Tensor>& tensors);
std::vector<c10::optional<at::Tensor>> GetMetaOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors);
} // namespace habana

#define HPU_SUPPORTED_DTYPES(fn, supported_dtypes)                       \
  const static std::unordered_set<c10::ScalarType> fn##_supported_dtypes \
      supported_dtypes;

#define FALLBACK_IF_UNSUPPORTED_DTYPE(tensor, fn, args...)       \
  if (ABSL_PREDICT_FALSE(                                        \
          tensor.defined() &&                                    \
          !fn##_supported_dtypes.count(tensor.scalar_type()))) { \
    return AtenHpuTypeDefault::fn(args);                         \
  }
