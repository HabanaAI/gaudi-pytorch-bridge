/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include <absl/functional/any_invocable.h>
#include "backend/habana_operator.h"

#pragma once

namespace habana {

template <class T>
using sizes_vec_template = std::vector<std::vector<T>>;

using sizes_vec = sizes_vec_template<int64_t>;
using sym_sizes_vec = sizes_vec_template<c10::SymInt>;

struct NodeAttr {
  struct NodeOutputAttr {
    at::IntArrayRef sizes{};
    at::ScalarType dtype{at::kFloat};
    c10::optional<int> final_result_index{c10::nullopt};
    synTensorType tensor_type{DATA_TENSOR};
    synDataType syn_data_type{syn_type_na};
    c10::optional<std::variant<synapse_helpers::tensor*, int>> inplace_out_ptr{
        c10::nullopt};
    c10::optional<unsigned> exp_bias{c10::nullopt};
  };

  std::string guid;
  std::vector<synTensor> inputs;
  std::vector<NodeOutputAttr> output_attrs;
  void* params = nullptr;
  size_t param_size = 0;
  std::string inf_name = std::string();
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
  bool isOutputInfMode() const {
    return m_output_inf_mode;
  }

  InferOutputMetaRetType& GetOutputInfMeta() {
    return m_output_inf_meta;
  }

  const c10::ScalarType& ScalarType() const {
    return m_scalar_type;
  }

  void SetScalarType(c10::ScalarType dtype) {
    m_scalar_type = dtype;
  }

  const auto& GetShapeTensors() const {
    return m_shape_tensors;
  }

  void CreateShapeTensorInput(
      synapse_helpers::graph& graph,
      at::ScalarType dtype,
      at::IntArrayRef sizes,
      std::vector<synTensor>& inputs,
      synTensorType shape_tensor_type = SHAPE_TENSOR,
      bool force_create = false,
      void* hostDataPtr = nullptr);

  void CreateH2dTensorInput(
      synapse_helpers::graph& graph,
      at::ScalarType dtype,
      void* hostDataPtr,
      size_t hostDataSize,
      std::vector<synTensor>& inputs,
      synTensorType shape_tensor_type = HOST_TO_DEVICE_TENSOR,
      bool force_create = false);

  sizes_vec ComputeOutputShapes(const at::Stack& stack) const {
    if (m_compute_output_shapes) {
      return m_compute_output_shapes(stack);
    }
    return {};
  }

  OutputMetaDataVector OutputMeta(const at::Stack& stack) const;
  bool STMeta(
      habana_helpers::IShapeList& inputs,
      habana_helpers::IShapeList& outputs) const;
  void HandleScalarToTensorSTMeta(habana_helpers::IShapeList& inputs) const;
  void SetOutputMetadata(OutputMetaDataVector meta_vec) {
    m_output_metadata = std::move(meta_vec);
  }

  const synapse_helpers::tensor_or_ref& ReadSynInput(size_t index);

  int GetNumSynNodes() const {
    return m_num_syn_nodes;
  }

  const OutputMetaDataVector& GetOutputMetaData() const {
    return m_output_metadata;
  }

 protected:
  std::vector<int> ScalarId() const {
    return m_scalar_ids;
  }

  bool IsOutputAvailable() const {
    return m_is_outfn or m_inplace_ids.size();
  }

  bool IsOutputPersistent(int i) const {
    return m_output_metadata.at(i).persistent;
  }

  bool IsInplace() const {
    return m_inplace_ids.size();
  }

  void SetSynapseLayouts(
      std::vector<synapse_helpers::layouts::SynapseLayoutFormat> in_layouts,
      std::vector<synapse_helpers::layouts::SynapseLayoutFormat> out_layouts) {
    kernel_meta_data_.synapse_input_layout = std::move(in_layouts);
    kernel_meta_data_.synapse_output_layout = std::move(out_layouts);
  }

  void SetTpcInputOrder(const std::vector<std::size_t>& tpc_input_order) {
    kernel_meta_data_.tpc_input_order = tpc_input_order;
  }

  void SetNumOutTensors(int n) {
    m_num_out_tensors = n;
  }

  void EnableTypePromotion() {
    m_promote_type = true;
  }

  void HandleBoolInputs() {
    m_cast_bool_to_uint8 = true;
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

  void SetComputeOutputShapes(std::function<sizes_vec(const at::Stack&)> fn) {
    m_compute_output_shapes = std::move(fn);
  }

  void SetOutputMetaFn(
      std::function<OutputMetaDataVector(const at::Stack&)> fn) {
    m_output_meta_fn = std::move(fn);
  }

  void SetOutputMetaFn(
      std::function<PartialOutputMetaDataVector(const at::Stack&)> fn) {
    m_partial_output_meta_fn = std::move(fn);
  }

  void SetSharedLayerMetaFn(
      std::function<SharedMetaDataVector(const at::Stack&)> fn) {
    m_shared_layer_meta_fn = std::move(fn);
  }

  void SetSTMetaFn(std::function<bool(
                       habana_helpers::IShapeList& inputs,
                       habana_helpers::IShapeList& outputs)> fn) {
    m_st_meta_fn = std::move(fn);
  }

  const OutputMetaData& GetOutputMetaData(int i) const {
    return m_output_metadata.at(i);
  }

  void SetHwScalingIds(const std::vector<int>& ids) {
    m_hw_scaling_ids = ids;
  }

  bool UsesOutputMeta() const {
    return m_output_meta_fn or m_partial_output_meta_fn;
  }

  void AddUndefinedOutputTensor() {
    auto& meta = GetOutputInfMeta();
    meta.AddUndefinedOutputTensor();
  }

  virtual void CustomHandler(synapse_helpers::graph&, at::Stack&) {}

 private:
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      at::Stack& stack,
      const OutputMetaDataVector& output_metadata) override;

  void PopulateMetadata(const at::Stack&, const OutputMetaDataVector&);
  void HandleScalarToTensor(
      synapse_helpers::graph& graph,
      const at::Stack& stack);
  void HandleFn(synapse_helpers::graph& graph);
  void HandleInplaceFn(synapse_helpers::graph& graph, const at::Stack& stack);
  void HandleOutFn(synapse_helpers::graph& graph, const at::Stack& stack);
  void HandleTypePromotion(
      synapse_helpers::graph& graph,
      const at::Stack& stack);
  void HandleHwScaling(const at::Stack&, const size_t);

  static synapse_helpers::tensor BuildRegularCast(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      const at::IntArrayRef sizes,
      const at::ScalarType& from,
      const at::ScalarType& to,
      c10::optional<int> final_result_index);

 protected:
  std::vector<synapse_helpers::tensor> BuildOp(
      synapse_helpers::graph& graph,
      std::string guid,
      std::vector<synTensor>&& node_inputs,
      std::vector<NodeAttr::NodeOutputAttr> node_output_attr,
      void* params = nullptr,
      size_t param_size = 0,
      std::string name = std::string());

  synTensor syn_in(size_t index);
  synapse_helpers::tensor& syn_out(size_t index);
  synapse_helpers::tensor_or_ref& SynInput(size_t index) override;
  void EraseSynInput(int index);
  synTensor syn_seed();

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
      c10::optional<int> final_result_index = c10::nullopt,
      c10::optional<unsigned> exp_bias = c10::nullopt);

  synapse_helpers::tensor BroadcastHelper(
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt,
      c10::optional<unsigned> exp_bias = c10::nullopt);

  synapse_helpers::tensor PermuteHelper(
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::IntArrayRef permutation,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt,
      c10::optional<unsigned> exp_bias = c10::nullopt);

  synapse_helpers::tensor IdentityHelper(
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt,
      c10::optional<unsigned> exp_bias = c10::nullopt);

  synapse_helpers::tensor SqueezeHelper(
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<unsigned> axis = c10::nullopt,
      c10::optional<int> final_result_index = c10::nullopt);

  synapse_helpers::tensor ExpandDimsHelper(
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      unsigned axis,
      c10::optional<int> final_result_index = c10::nullopt);

  synapse_helpers::tensor FlattenHelper(
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt);

  virtual void AddNode(synapse_helpers::graph&, const at::Stack&);

 public:
  InferOutputMetaRetType InferOutputMeta(at::Stack&) override;

  static std::vector<synapse_helpers::tensor> BuildNode(
      OpBackend* op,
      synapse_helpers::graph& graph,
      NodeAttr&& node_attr);

  static synapse_helpers::tensor BuildCast(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      const at::IntArrayRef sizes,
      const at::ScalarType& from,
      const at::ScalarType& to,
      c10::optional<int> final_result_index = c10::nullopt);

  static synapse_helpers::tensor BuildBoolCast(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      const at::IntArrayRef sizes,
      const at::ScalarType& from,
      c10::optional<int> final_result_index = c10::nullopt);

  static synapse_helpers::tensor BuildConstant(
      OpBackend* op,
      synapse_helpers::graph& graph,
      const at::Scalar& val,
      c10::optional<at::ScalarType> force_type = c10::nullopt,
      const at::IntArrayRef constant_outshape = 1,
      c10::optional<int> final_result_index = c10::nullopt);

  static synapse_helpers::tensor BuildConstantTensor(
      OpBackend* op,
      synapse_helpers::graph& graph,
      const at::Scalar& val,
      c10::optional<at::ScalarType> force_type = c10::nullopt,
      const at::IntArrayRef constant_outshape = 1);

  static synapse_helpers::tensor BuildReshape(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt,
      c10::optional<unsigned> exp_bias = c10::nullopt);

  static synapse_helpers::tensor BuildBroadcast(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt,
      c10::optional<unsigned> exp_bias = c10::nullopt);

  static synapse_helpers::tensor BuildIdentity(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt,
      c10::optional<unsigned> exp_bias = c10::nullopt);

  static synapse_helpers::tensor BuildSqueeze(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<unsigned> axis = c10::nullopt,
      c10::optional<int> final_result_index = c10::nullopt);

  static synapse_helpers::tensor BuildExpandDims(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      unsigned axis,
      c10::optional<int> final_result_index = c10::nullopt);

  static synapse_helpers::tensor BuildFlatten(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt);

  static std::vector<synapse_helpers::tensor> BuildNonZero(
      OpBackend*,
      synapse_helpers::graph&,
      synapse_helpers::tensor&,
      at::IntArrayRef,
      at::ScalarType,
      c10::optional<int> = c10::nullopt);

  static synapse_helpers::tensor BuildScatterNDOnnx(
      OpBackend*,
      synapse_helpers::graph&,
      const std::vector<synTensor>&,
      at::IntArrayRef,
      at::ScalarType,
      c10::optional<int> = c10::nullopt);

  static synapse_helpers::tensor BuildPermute(
      OpBackend* op,
      synapse_helpers::graph& graph,
      synTensor syn_in,
      at::IntArrayRef sizes,
      at::IntArrayRef permutation,
      at::ScalarType dtype,
      c10::optional<int> final_result_index = c10::nullopt,
      c10::optional<unsigned> exp_bias = c10::nullopt);

  struct TensorsPair {
    const at::Tensor& pt_t;
    synTensor syn_t;
    int syn_idx = -1; // In case it's needed to call SynInput instead of syn_in,
                      // we hold also syn_idx
  };

 protected:
  // The class StackGetter and this::getNextInputInternal() overloads are
  // separate as we have to call SynInput() base class member. It is not
  // accessible from StackGetter class members.
  class StackGetter {
   public:
    StackGetter(const at::Stack& stackIn, const char* labelIn)
        : stack(stackIn), label(labelIn) {}

    size_t CheckGetAndIncrStackPos() {
      TORCH_CHECK(
          stackPos < stack.size(),
          label,
          " expected at least ",
          stackPos + 1,
          " args on stack but got ",
          stack.size())
      return stackPos++;
    }

    size_t GetAndIncrSynPos() {
      return synPos++;
    }

    void MoveSynPos(size_t offset) {
      synPos += offset;
    }

    const at::Stack& stack;

   private:
    size_t stackPos = 0;
    size_t synPos = 0;
    const char* label;
  };

  template <class T>
  auto getNextInput(StackGetter& sg) {
    return getNextInputInternal(sg, (T*){});
  }

 private:
  c10::IValue getNextInputInternal(StackGetter& sg, c10::IValue*) {
    auto pos = sg.CheckGetAndIncrStackPos();
    if (sg.stack[pos].isTensor()) {
      sg.MoveSynPos(1);
    } else if (sg.stack[pos].isTensorList()) {
      sg.MoveSynPos(sg.stack[pos].toTensorList().size());
    }
    return sg.stack[pos];
  }

  TensorsPair getNextInputInternal(StackGetter& sg, TensorsPair*) {
    auto pos = sg.CheckGetAndIncrStackPos();
    TORCH_CHECK(
        sg.stack[pos].isTensor(),
        "Input ",
        pos,
        " type expected to be ",
        "tensor");
    int syn_pos = sg.GetAndIncrSynPos();
    return {sg.stack[pos].toTensor(), syn_in(syn_pos), syn_pos};
  }

  c10::optional<TensorsPair> getNextInputInternal(
      StackGetter& sg,
      c10::optional<TensorsPair>*) {
    auto pos = sg.CheckGetAndIncrStackPos();
    TORCH_CHECK(
        sg.stack[pos].isNone() || sg.stack[pos].isTensor(),
        "Input ",
        pos,
        " type expected to be ",
        "none or tensor");
    if (sg.stack[pos].isTensor()) {
      int syn_pos = sg.GetAndIncrSynPos();
      return TensorsPair{sg.stack[pos].toTensor(), syn_in(syn_pos), syn_pos};
    } else {
      return c10::optional<TensorsPair>{};
    }
  }

  c10::optional<std::vector<TensorsPair>> getNextInputInternal(
      StackGetter& sg,
      c10::optional<std::vector<TensorsPair>>*) {
    auto pos = sg.CheckGetAndIncrStackPos();
    c10::optional<c10::List<at::Tensor>> tensorList =
        sg.stack[pos].toOptional<c10::List<at::Tensor>>();
    if (tensorList.has_value()) {
      c10::optional<std::vector<TensorsPair>> result =
          std::vector<TensorsPair>();
      for (auto&& v : tensorList.value()) {
        result.value().push_back({v, syn_in(sg.GetAndIncrSynPos())});
      }
      return result;
    } else {
      return c10::optional<std::vector<TensorsPair>>{};
    }
  }

  std::vector<TensorsPair> getNextInputInternal(
      StackGetter& sg,
      std::vector<TensorsPair>*) {
    auto pos = sg.CheckGetAndIncrStackPos();
    TORCH_CHECK(
        sg.stack[pos].isTensorList(),
        "Input ",
        pos,
        " type expected to be ",
        "tensor list");
    auto list = sg.stack[pos].toTensorList();
    std::vector<TensorsPair> result;
    for (auto&& v : list) {
      result.push_back({v, syn_in(sg.GetAndIncrSynPos())});
    }
    return result;
  }

  c10::optional<c10::ScalarType> getNextInputInternal(
      StackGetter& sg,
      c10::optional<c10::ScalarType>*) {
    auto pos = sg.CheckGetAndIncrStackPos();
    TORCH_CHECK(
        sg.stack[pos].isNone() || sg.stack[pos].isInt(),
        "Input ",
        pos,
        " type expected to be ",
        "none or ScalarType");
    return sg.stack[pos].toOptional<at::ScalarType>();
  }

  std::variant<TensorsPair, c10::IValue> getNextInputInternal(
      StackGetter& sg,
      std::variant<TensorsPair, c10::IValue>*) {
    auto pos = sg.CheckGetAndIncrStackPos();
    if (sg.stack[pos].isTensor()) {
      int syn_pos = sg.GetAndIncrSynPos();
      return TensorsPair{sg.stack[pos].toTensor(), syn_in(syn_pos)};
    } else {
      return sg.stack[pos];
    }
  }

#define GET_NEXT_INPUT_INTERNAL(T, isFn, toFn, Tstr)                         \
  T getNextInputInternal(StackGetter& sg, T*) {                              \
    auto pos = sg.CheckGetAndIncrStackPos();                                 \
    TORCH_CHECK(                                                             \
        sg.stack[pos].isFn(), "Input ", pos, " type expected to be ", Tstr); \
    return sg.stack[pos].toFn();                                             \
  }

  GET_NEXT_INPUT_INTERNAL(bool, isBool, toBool, "bool")
  GET_NEXT_INPUT_INTERNAL(double, isDouble, toDouble, "double")
  GET_NEXT_INPUT_INTERNAL(int, isInt, toInt, "int")
  GET_NEXT_INPUT_INTERNAL(c10::List<bool>, isBoolList, toBoolList, "bool array")
  GET_NEXT_INPUT_INTERNAL(c10::ScalarType, isInt, toScalarType, "ScalarType")
  GET_NEXT_INPUT_INTERNAL(
      std::vector<int64_t>,
      isIntList,
      toIntList().vec,
      "int list")
  GET_NEXT_INPUT_INTERNAL(c10::string_view, isString, toStringView, "string")
#undef GET_NEXT_INPUT_INTERNAL

 private:
  const std::vector<int> m_res_ids;
  const std::vector<int> m_inplace_ids;
  const std::vector<int> m_scalar_ids;
  const bool m_is_outfn;

  c10::ScalarType m_scalar_type;
  std::optional<int> m_output_type_stack_idx;
  bool m_promote_type = false;
  bool m_cast_bool_to_uint8 = false;
  bool m_promote_int_to_float = false;
  int m_num_out_tensors = 1;
  std::vector<int> m_hw_scaling_ids;

  // For shape inference of outputs/intermediates
  bool m_output_inf_mode = false;
  InferOutputMetaRetType m_output_inf_meta;
  int m_num_syn_nodes = 0;

  std::function<std::shared_ptr<void>(const at::Stack&, size_t&)> m_fill_params;
  std::function<sizes_vec(const at::Stack&)> m_compute_output_shapes;
  std::function<OutputMetaDataVector(const at::Stack&)> m_output_meta_fn;
  std::function<PartialOutputMetaDataVector(const at::Stack&)>
      m_partial_output_meta_fn;
  std::function<SharedMetaDataVector(const at::Stack&)> m_shared_layer_meta_fn;
  std::function<bool(
      habana_helpers::IShapeList& inputs,
      habana_helpers::IShapeList& outputs)>
      m_st_meta_fn;
  std::vector<synapse_helpers::tensor> m_shape_tensors;

  std::unordered_map<size_t, synapse_helpers::tensor_or_ref> syn_inputs_cast_;

  OutputMetaDataVector m_output_metadata;

  // Those are lambdas that shall be executed after
  // node has been added in AllocateAndAddSynapseNode.
  // They are utilised ba HandleOutFn to insert cast
  // before output tensor provided by user
  // and the result of executed kernel
  // as there might be some discrepancies
  // between kernel supported type and user tensor type.
  std::vector<absl::AnyInvocable<void()>> m_post_add_node_functions;
};
} // namespace habana
