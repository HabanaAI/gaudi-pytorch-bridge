/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/ops/unpack.h"

using namespace at;
using namespace habana;

namespace habana_lazy {

template <typename ReturnType, typename NodeConstruct = void>
class LazyOptimizationOp : public LazyOp<ReturnType> {
 public:
  explicit LazyOptimizationOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::set<size_t>& metadata_indices = {},
      const std::vector<std::vector<int64_t>>& out_shapes = {},
      int out_index = 0)
      : LazyOp<ReturnType>(
            qualstring,
            inputs,
            metadata_indices,
            out_shapes,
            out_index) {}

  virtual ~LazyOptimizationOp() = default;

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList& tList1) {
    auto context = habana_lazy_executor.getDeviceExecutionContext();
    const auto& node = LazyOp<T>::create_node();

    const auto noOfTensor = tList1.size();

    int64_t out_index = 0;
    for (size_t i = 0; i < noOfTensor; ++i) {
      auto hl_result1 = GetHbLazyTensor(tList1[i]);

      updateDstDependencies(hl_result1, tList1[i], true);

      ir::Value& out1 = hl_result1.CurrentIrValue();
      out1.SetNode(
          node,
          hl_result1.GetDevice(),
          hl_result1.GetSizes(),
          hl_result1.dtype_optional(),
          out_index++);

      context->MarkTensorStatus(
          hl_result1.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
    }
    LazyOp<T>::runSBS(tList1);
    flush_op(tList1);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList& tList1,
      at::TensorList& tList2) {
    auto context = habana_lazy_executor.getDeviceExecutionContext();
    const auto& node = LazyOp<T>::create_node();

    const auto noOfTensor = tList1.size();
    int64_t out_index = 0;
    for (size_t i = 0; i < noOfTensor; ++i) {
      auto hl_result1 = GetHbLazyTensor(tList1[i]);
      auto hl_result2 = GetHbLazyTensor(tList2[i]);

      updateDstDependencies(hl_result1, tList1[i], true);
      updateDstDependencies(hl_result2, tList2[i], true);

      ir::Value& out1 = hl_result1.CurrentIrValue();
      out1.SetNode(
          node,
          hl_result1.GetDevice(),
          hl_result1.GetSizes(),
          hl_result1.dtype_optional(),
          out_index++);

      ir::Value& out2 = hl_result2.CurrentIrValue();
      out2.SetNode(
          node,
          hl_result2.GetDevice(),
          hl_result2.GetSizes(),
          hl_result2.dtype_optional(),
          out_index++);

      context->MarkTensorStatus(
          hl_result1.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
      context->MarkTensorStatus(
          hl_result2.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
    }
    LazyOp<T>::runSBS(tList1);
    LazyOp<T>::runSBS(tList2);
    flush_op(tList1);
    flush_op(tList2);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList& tList1,
      at::TensorList& tList2,
      at::TensorList& tList3,
      const float flag) {
    auto context = habana_lazy_executor.getDeviceExecutionContext();
    const auto& node = LazyOp<T>::create_node();

    const auto noOfTensor = tList1.size();
    int64_t out_index = 0;

    auto hlList5 = habana_lazy::GetHbLazyTensor(tList3[0]);
    habana_lazy::ir::Value& out = hlList5.CurrentIrValue();
    node->set_as_output_tensor_list();
    out.SetNode(
        node,
        hlList5.GetDevice(),
        hlList5.GetSizes(),
        hlList5.dtype_optional());

    habana_lazy::ir::NodePtr node_unpack =
        std::make_shared<habana_lazy::ir::ListUnpack>(out);

    for (size_t i = 0; i < noOfTensor; ++i) {
      auto hl_result1 = GetHbLazyTensor(tList1[i]);
      auto hl_result2 = GetHbLazyTensor(tList1[i]);
      auto hl_result3 = GetHbLazyTensor(tList2[i]);
      auto hl_result4 = GetHbLazyTensor(tList2[i]);
      auto hl_result5 = GetHbLazyTensor(tList3[i]);

      if (1.0 != flag) {
        auto hl_list3 = GetHbLazyTensor(tList3[i]);
        ir::Value& out0 = hl_list3.CurrentIrValue();
        out0.SetNode(
            node_unpack,
            hl_list3.GetDevice(),
            hl_list3.GetSizes(),
            hl_list3.dtype_optional(),
            out_index++);
      }

      ir::Value& out1 = hl_result1.CurrentIrValue();
      out1.SetNode(
          node,
          hl_result1.GetDevice(),
          hl_result1.GetSizes(),
          hl_result1.dtype_optional(),
          out_index++);

      ir::Value& out2 = hl_result2.CurrentIrValue();
      out2.SetNode(
          node,
          hl_result2.GetDevice(),
          hl_result2.GetSizes(),
          hl_result2.dtype_optional(),
          out_index++);

      ir::Value& out3 = hl_result3.CurrentIrValue();
      out3.SetNode(
          node,
          hl_result3.GetDevice(),
          hl_result3.GetSizes(),
          hl_result3.dtype_optional(),
          out_index++);

      ir::Value& out4 = hl_result4.CurrentIrValue();
      out4.SetNode(
          node,
          hl_result4.GetDevice(),
          hl_result4.GetSizes(),
          hl_result4.dtype_optional(),
          out_index++);

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
    flush_op(tList1);
    flush_op(tList2);
    flush_op(tList3);
  }
}; // class LazyOptimizationOp

} // namespace habana_lazy
