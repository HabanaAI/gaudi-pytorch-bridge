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
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/lazy_graph_hash_builder.h"
#include "habana_lazy/ops/unpack.h"

namespace habana_lazy {

enum OPTIMIZER { ADAGRAD = 0, SGD_MOMENTUM, LARS, OTHER, NO_OF_OPTIMIZER };

template <typename ReturnType, typename NodeConstruct = void>
class LazyOptimizationOp : public LazyOp<ReturnType> {
 public:
  explicit LazyOptimizationOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::vector<std::vector<int64_t>>& out_shapes = {},
      int out_index = 0)
      : LazyOp<ReturnType>(qualstring, inputs, out_shapes, out_index) {}

  virtual ~LazyOptimizationOp() = default;

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      std::vector<at::Tensor>& tVector) {
    LazyOp<T>::viewUpdateInputs();
    const auto& node = LazyOp<T>::create_node();

    const auto noOfTensor = tVector.size();

    int64_t out_index = 0;
    for (size_t i = 0; i < noOfTensor; ++i) {
      HbLazyTensorViews::CustomKernelAddNodeInplace(
          tVector[i], node, out_index);
    }
    flush_op();
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList& tList1,
      [[maybe_unused]] OPTIMIZER optimizer = OTHER) {
    LazyOp<T>::viewUpdateInputs();
    const auto& node = LazyOp<T>::create_node();

    const auto noOfTensor = tList1.size();

    int64_t out_index = 0;
    auto t = GetHbLazyTensor(tList1[0]);
    node->set_as_output_tensor_list();
    ir::Value& out{t.IrSetNode(node)};

    ir::NodePtr node_unpack = std::make_shared<ir::ListUnpack>(out);

    for (size_t i = 0; i < noOfTensor; ++i) {
      HbLazyTensorViews::CustomKernelAddNodeInplace(
          tList1[i], node_unpack, out_index);
    }
    flush_op({});
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList& tList1,
      at::TensorList& tList2,
      enum OPTIMIZER optimizer) {
    if (SGD_MOMENTUM == optimizer) {
      callSGD_momentum(tList1, tList2);
    } else if (ADAGRAD == optimizer) {
      callAdagrad(tList1, tList2);
    } else {
      TORCH_CHECK(
          false,
          "Incorrect optmizer option. Only ADAGRAD/SGD_MOMENTUM can be called with 2 at::TensorList& arguments.")
    }
    flush_op();
  }
  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList& tList1,
      at::TensorList& tList2,
      at::TensorList& tList3,
      // Fetch first output if True else skip
      // This is required for adamw where for
      // modified-weight-decay !=1,
      // we send "weights" as first output component.
      const bool flagAdditionalOutput) {
    LazyOp<T>::viewUpdateInputs();
    auto context = get_device_lazy_execution_context();
    const auto& node = LazyOp<T>::create_node();

    const auto noOfTensor = tList1.size();
    int64_t out_index = 0;

    auto hlList5 = habana_lazy::GetHbLazyTensor(tList3[0]);
    node->set_as_output_tensor_list();
    auto& out{hlList5.IrSetNode(node)};

    habana_lazy::ir::NodePtr node_unpack =
        std::make_shared<habana_lazy::ir::ListUnpack>(out);

    for (size_t i = 0; i < noOfTensor; ++i) {
      auto hl_result1 = GetHbLazyTensor(tList1[i]);
      auto hl_result2 = GetHbLazyTensor(tList1[i]);
      auto hl_result3 = GetHbLazyTensor(tList2[i]);
      auto hl_result4 = GetHbLazyTensor(tList2[i]);
      auto hl_result5 = GetHbLazyTensor(tList3[i]);

      if (flagAdditionalOutput) {
        auto hl_list3 = GetHbLazyTensor(tList3[i]);
        hl_list3.IrSetNode(node_unpack, out_index++);
      }

      hl_result1.IrSetNode(node, out_index++);

      hl_result2.IrSetNode(node, out_index++);

      hl_result3.IrSetNode(node, out_index++);

      hl_result4.IrSetNode(node, out_index++);

      HbLazyTensorViews::CustomKernelAddNodeInplace(
          tList3[i], node_unpack, out_index);

      context->MarkTensorStatus(
          hl_result1.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
      context->MarkTensorStatus(
          hl_result2.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
      context->MarkTensorStatus(
          hl_result3.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
      context->MarkTensorStatus(
          hl_result4.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
    }
    LazyOp<T>::runSBS(tList1);
    LazyOp<T>::runSBS(tList2);
    LazyOp<T>::runSBS(tList3);
    flush_op(tList1.size() + tList2.size() + tList3.size());
  }

 private:
  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type callSGD_momentum(
      at::TensorList& tList1,
      at::TensorList& tList2) {
    LazyOp<T>::viewUpdateInputs();
    auto context = get_device_lazy_execution_context();
    const auto& node = LazyOp<T>::create_node();
    const auto noOfTensor = tList1.size();
    int64_t out_index = 0;

    auto hl_result1 = GetHbLazyTensor(tList1[0]);
    node->set_as_output_tensor_list();
    ir::Value& out{hl_result1.IrSetNode(node)};
    habana_lazy::ir::NodePtr node_unpack =
        std::make_shared<habana_lazy::ir::ListUnpack>(out);
    for (size_t i = 0; i < noOfTensor; ++i) {
      HbLazyTensorViews::CustomKernelAddNodeInplace(
          tList1[i], node_unpack, out_index);

      auto hl_result2 = GetHbLazyTensor(tList2[i]);
      hl_result2.IrSetNode(node_unpack, out_index++);

      context->MarkTensorStatus(
          hl_result1.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
      context->MarkTensorStatus(
          hl_result2.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
    }
    LazyOp<T>::runSBS(tList1);
    LazyOp<T>::runSBS(tList2);
    flush_op(tList1.size() + tList2.size());
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type callAdagrad(
      at::TensorList& tList1,
      at::TensorList& tList2) {
    LazyOp<T>::viewUpdateInputs();
    auto context = get_device_lazy_execution_context();
    const auto& node = LazyOp<T>::create_node();

    const auto noOfTensor = tList1.size();
    int64_t out_index = 0;
    for (size_t i = 0; i < noOfTensor; ++i) {
      auto hl_result1 = GetHbLazyTensor(tList1[i]);
      auto hl_result2 = GetHbLazyTensor(tList2[i]);

      HbLazyTensorViews::CustomKernelAddNodeInplace(tList1[i], node, out_index);

      hl_result2.IrSetNode(node, out_index++);

      context->MarkTensorStatus(
          hl_result1.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
      context->MarkTensorStatus(
          hl_result2.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
    }
    LazyOp<T>::runSBS(tList1);
    LazyOp<T>::runSBS(tList2);
    flush_op(tList1.size() + tList2.size());
  }

}; // class LazyOptimizationOp

} // namespace habana_lazy
