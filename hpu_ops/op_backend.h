/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_kernels/habana_operator.h"

#pragma once
namespace habana {

using sizes_vec = std::vector<std::vector<int64_t>>;

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
  bool isMetaMode() const {
    return m_meta_mode;
  }

  OutputShapeInfRetType& GetMeta() {
    return m_meta;
  }

  const c10::ScalarType& ScalarType() const {
    return m_scalar_type;
  }

  void SetScalarType(c10::ScalarType dtype) {
    m_scalar_type = dtype;
  }

  // keeping AllocateAndAddSynapseNode public to help with calling autogen ops
  // from manually written ops
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      at::Stack& stack,
      const OutputMetaDataVector& output_metadata) override;

  const auto& GetShapeTensors() const {
    return m_shape_tensors;
  }

  const synapse_helpers::tensor& CreateShapeTensorInput(
      synapse_helpers::graph& graph,
      at::ScalarType dtype,
      at::IntArrayRef sizes,
      synTensorType shape_tensor_type);

  sizes_vec ComputeOutputShapes(
      const at::Stack& stack,
      bool is_lowering = false) const {
    if (m_compute_output_shapes) {
      return m_compute_output_shapes(stack, is_lowering);
    }
    return {};
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
    return m_output_metadata.at(i).persistent;
  }

  const OutputMetaData& GetOutputMetaData(int i) const {
    return m_output_metadata.at(i);
  }

  bool IsInplace() const {
    return m_inplace_ids.size();
  }

  void SetLayouts(
      std::vector<LayoutFormat> in_layouts,
      std::vector<LayoutFormat> out_layouts) {
    kernel_meta_data_.input_layout = std::move(in_layouts);
    kernel_meta_data_.output_layout = std::move(out_layouts);
  }

  void SetSynapseLayouts(
      std::vector<synapse_helpers::layouts::SynapseLayoutFormat> in_layouts,
      std::vector<synapse_helpers::layouts::SynapseLayoutFormat> out_layouts) {
    kernel_meta_data_.synapse_input_layout = std::move(in_layouts);
    kernel_meta_data_.synapse_output_layout = std::move(out_layouts);
  }

  void SetNumOutTensors(int n) {
    m_num_out_tensors = n;
  }

  void EnableTypePromotion() {
    m_promote_type = true;
  }

  bool IsTypePromotion() const {
    return m_promote_type;
  }

  void PromoteIntToFloat() {
    m_promote_int_to_float = true;
  }

  bool IsPromoteIntToFloat() const {
    return m_promote_int_to_float;
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

 protected:
  std::vector<synapse_helpers::tensor> BuildOp(
      synapse_helpers::graph& graph,
      const std::string& guid,
      std::vector<synTensor> node_inputs,
      const std::vector<NodeAttr::NodeOutputAttr>& node_output_attr,
      void* params = nullptr,
      size_t param_size = 0);

  synTensor syn_in(int index);
  synapse_helpers::tensor& syn_out(int index);

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

  OutputShapeInfRetType ComputeOutputShape(at::Stack&) override;

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

  synapse_helpers::graph* m_graph = nullptr;

  // For shape inference of outputs/intermediates
  bool m_meta_mode = false;
  OutputShapeInfRetType m_meta;

  std::unordered_map<int, at::Scalar> m_scalar_inputs;
  std::function<std::shared_ptr<void>(const at::Stack&, size_t&)> m_fill_params;
  std::function<sizes_vec(const at::Stack&, bool)> m_compute_output_shapes;
  std::vector<synapse_helpers::tensor> m_shape_tensors;

 protected:
  OutputMetaDataVector m_output_metadata;
};
} // namespace habana
