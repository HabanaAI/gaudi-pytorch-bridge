/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#pragma once

#include <c10_ver/core/SymIntArrayRef.h>
#include <tuple>
#include <utility>

#include "backend/helpers/tensor_utils.h"
#include "backend/jit_graph_cache.h"
#include "backend/synapse_helpers/env_flags.h"
#include "common/list_of_lists_custom_iterator.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/template_helpers.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/lazy_graph_hash_builder.h"
#include "habana_lazy/sbs_runner.h"
#include "habana_lazy/view_utils.h"
#include "lazy_kernels_declarations.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"
#include "resize.h"

#include "habana_lazy/memlog.h"
#include "habana_lazy/ops/shape_ops.h"

namespace habana_lazy {
at::Tensor append_to_batch_h2d_list(const at::Tensor& scalar_tensor);
void updateDstDependencies(const at::Tensor& dst);

at::Tensor empty_as_strided_lazy(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);

at::Tensor handleWeightTensorLayout(const at::Tensor& src);

ir::NodePtr create_as_strided_node(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    at::IntArrayRef orig_size,
    at::IntArrayRef orig_stride,
    c10::optional<int64_t> storage_offset,
    bool is_out = false);

ir::NodePtr create_as_strided_node(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset,
    bool is_out = false);

bool is_inplace(at::Symbol symbol);

void InitSizesAndStrides(
    at::Tensor& at_tensor,
    c10::optional<synTensorType> tensor_type,
    c10::optional<c10::IntArrayRef> size,
    c10::optional<c10::IntArrayRef> stride,
    c10::optional<c10::MemoryFormat> mem_format);
std::vector<int64_t> CalculateStrides(
    const c10::IntArrayRef sizes,
    c10::MemoryFormat format);
std::vector<int64_t> CalculateStrides5d(
    const c10::IntArrayRef sizes,
    c10::MemoryFormat format);

at::Tensor get_tensor_for_scalar(
    double alpha,
    const at::TensorOptions& options = {});

void flush_op(
    size_t out_tensor_count = 0,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info = nullptr,
    std::vector<HbLazyTensor> out_hb_lazy_tensor = {});

// TODO: Ideally we want a variant of HABANA_ASSERT like
// TORCH_INTERNAL_ASSERT_DEBUG_ONLY

void handle_collective(const at::IValue& value);
void handle_collective(const at::Tensor& tensor);
void handle_collective(const at::TensorList& list);
void handle_collective(const std::vector<at::Tensor>& vec);
void handle_collective(const at::ITensorListRef& list);

inline bool lazyEagerOptimizedViewHandling() {
  return (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING);
}

template <typename ReturnType, typename NodeConstruct = void>
class LazyOp {
 public:
  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{out_index},
        m_sbs_runner{SBSInterface::getSBSHandler(m_symbol.toQualString())},
        m_collective_op(habana_helpers::IsCollective(m_symbol)) {
    module_name = *(habana_lazy::ir::getCurrentModuleName());
    set_inputs(inputs);
  }

  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::function<std::vector<std::vector<int64_t>>(const at::Stack&)>&
          out_shapes_fn,
      int out_index = 0) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_out_index{out_index},
        m_sbs_runner{SBSInterface::getSBSHandler(m_symbol.toQualString())},
        m_collective_op(habana_helpers::IsCollective(m_symbol)) {
    module_name = *(habana_lazy::ir::getCurrentModuleName());
    if (out_shapes_fn) {
      m_out_shapes = out_shapes_fn(inputs);
    }
    set_inputs(inputs);
  }

  explicit LazyOp(
      ir::NodePtr node,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0)
      : m_node{std::move(node)},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{out_index},
        m_sbs_runner{SBSInterface::getSBSHandler(
            m_node ? m_node->op().toQualString() : "")},
        m_collective_op(habana_helpers::IsCollective(m_symbol)) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        std::is_class<NodeConstruct>::value,
        "This constructor is valid only when NodeConstruct is a class.");
    module_name = *(habana_lazy::ir::getCurrentModuleName());
    set_inputs(inputs);
  }

  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const at::TensorList& output_meta_tensors) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_out_index{},
        m_out_meta_tensors{output_meta_tensors},
        m_sbs_runner{SBSInterface::getSBSHandler(m_symbol.toQualString())},
        m_collective_op(habana_helpers::IsCollective(m_symbol)) {
    set_inputs(inputs);
    module_name = *(habana_lazy::ir::getCurrentModuleName());
    for (const auto& out : m_out_meta_tensors) {
      m_out_shapes.emplace_back(out.sizes().vec());
    }
  }

  LazyOp(LazyOp&) = default;
  LazyOp(const LazyOp&) = default;
  LazyOp(LazyOp&&) = default;
  LazyOp& operator=(const LazyOp&) = default;
  LazyOp& operator=(LazyOp&) = default;
  LazyOp& operator=(LazyOp&&) = default;

  virtual ~LazyOp() = default;

  template <typename T = ReturnType>
  typename std::enable_if<not is_tuple_of_tensor_ref<T>::value, T>::type
  HandleLazy(
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend =
          nullptr) {
    bool isOptimizedLazyEager = false;
    if (info_to_lazy_backend) {
      isOptimizedLazyEager =
          info_to_lazy_backend->get_is_optimized_lazy_eager();
    }

    auto results = get_result();
    int i = 0;
    std::vector<at::Tensor> tensors;
    std::vector<HbLazyTensor> hl_results = {};
    bool is_collective = m_collective_op;
    tensors.reserve(std::tuple_size<T>::value);
    habana::for_each_in_tuple(
        results, [&hl_results, &tensors, &is_collective](const auto& result) {
          auto hl_result = GetHbLazyTensor(result, true, !is_collective);
          tensors.push_back(result);
          hl_results.push_back(hl_result);
        });

    if (isOptimizedLazyEager == false) {
      if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) &&
          info_to_lazy_backend) {
        // lazy eager - preparing the input tensor uids
        prepare_lazy_eager_input_uids(info_to_lazy_backend);

        handle_strided_inputs();
      }
      auto node = create_node();
      for (auto hl_result : hl_results) {
        hl_result.IrSetNode(node, i++);
      }
    } else {
      PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
      std::vector<ir::Value> input_vals = prepare_lazy_eager_input_values();
      info_to_lazy_backend->set_input_values(std::move(input_vals));
    }

    log_dev_mem_stats("Post-Accumulation", m_symbol.toQualString());
    runSBS(tensors);
    flush_op(tensors.size(), info_to_lazy_backend, hl_results);
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<not is_tuple_of_tensor_ref<T>::value, T>::type call() {
    PT_LAZY_DEBUG(
        "Lazy Call not_Tuple_Of_Tensor_ref :: ", m_symbol.toQualString());
    bool isView = false;
    habana_lazy::ir::setCurrentModuleName(module_name);
    if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2) ||
        !GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING)) {
      isView = viewUpdateInputs();
    }
    std::shared_ptr<HbLazyFrontEndInfoToBackend> infoToBackEnd =
        std::make_shared<HbLazyFrontEndInfoToBackend>();
    infoToBackEnd->set_lazy_op_name(m_symbol.toQualString());

    auto context = get_device_lazy_execution_context();

    if (is_optimized_lazy_eager_supported(
            isView, context->viewContext.isLazyViewPresent)) {
      size_t lazy_eager_key = 0;
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);
      infoToBackEnd->set_optimized_lazy_eager_key(lazy_eager_key);
      infoToBackEnd->set_is_optimized_lazy_eager(IsOptimizedLazyEagerCached);
    }

    context->viewContext.isLazyViewPresent = false;

    return HandleLazy(infoToBackEnd);
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensors<T>::value, T>::type call(
      T tensors) {
    PT_LAZY_DEBUG("Lazy Call Tuple_Of_Tensor :: ", m_symbol.toQualString());
    bool isView = false;
    habana_lazy::ir::setCurrentModuleName(module_name);
    if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2) ||
        !GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING)) {
      isView = viewUpdateInputs();
    }
    std::shared_ptr<HbLazyFrontEndInfoToBackend> infoToBackEnd =
        std::make_shared<HbLazyFrontEndInfoToBackend>();
    infoToBackEnd->set_lazy_op_name(m_symbol.toQualString());

    auto context = get_device_lazy_execution_context();

    if (is_optimized_lazy_eager_supported(
            isView, context->viewContext.isLazyViewPresent)) {
      size_t lazy_eager_key = 0;
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);
      infoToBackEnd->set_optimized_lazy_eager_key(lazy_eager_key);
      infoToBackEnd->set_is_optimized_lazy_eager(IsOptimizedLazyEagerCached);
    }

    context->viewContext.isLazyViewPresent = false;

    return HandleLazy(tensors, infoToBackEnd);
  }

  template <typename T = ReturnType>
  typename std::enable_if<
      (is_tuple_of_tensor_ref<T>::value || is_tuple_of_tensors<T>::value),
      T>::type
  HandleLazy(
      T results,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend =
          nullptr) {
    bool isOptimizedLazyEager = false;
    if (info_to_lazy_backend) {
      isOptimizedLazyEager =
          info_to_lazy_backend->get_is_optimized_lazy_eager();
    }
    int i = 0;
    std::vector<at::Tensor> tensors;
    std::vector<HbLazyTensor> hl_results = {};
    tensors.reserve(std::tuple_size<T>::value);
    auto context = get_device_lazy_execution_context();

    std::vector<std::vector<int64_t>> out_shapes;
    if (m_output_meta_fn) {
      const auto& meta = m_output_meta_fn(get_inputs());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          meta.size() == std::tuple_size<T>::value);
      for (const auto& output_meta : meta) {
        out_shapes.emplace_back(output_meta.shape);
      }
    } else {
      out_shapes = m_out_shapes;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          out_shapes.empty() || out_shapes.size() == std::tuple_size<T>::value);
    }

    habana::for_each_in_tuple(
        results,
        [&i, &hl_results, &tensors, context, out_shapes, this](
            const auto& result) {
          auto hl_result = GetHbLazyTensor(result, true, !m_collective_op);
          tensors.push_back(result);
          hl_results.push_back(hl_result);
          if (!out_shapes.empty()) {
            const auto& out_shape = out_shapes.at(i);
            if (result.sizes() != out_shape ||
                (!m_shape_was_changed_in_tuple.empty() &&
                 m_shape_was_changed_in_tuple[i])) {
              auto impl = hl_result.getAttachedTensorImpl();
              THHTensor_resizeNd(
                  impl, out_shape.size(), out_shape.data(), nullptr);
              result.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
            }
          }
          context->MarkTensorStatus(
              hl_result.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
          i++;
        });

    if (isOptimizedLazyEager == false) {
      if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) &&
          info_to_lazy_backend) {
        // lazy eager - preparing the input tensor uids
        prepare_lazy_eager_input_uids(info_to_lazy_backend);

        handle_strided_inputs();
      }
      i = 0;
      auto node = create_node();
      for (auto hl_result : hl_results) {
        hl_result.IrSetNode(node, i++);
      }
    } else {
      PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
      std::vector<ir::Value> input_vals = prepare_lazy_eager_input_values();
      info_to_lazy_backend->set_input_values(std::move(input_vals));
    }
    log_dev_mem_stats("Post-Accumulation", m_symbol.toQualString());
    runSBS(tensors);
    flush_op(tensors.size(), info_to_lazy_backend, hl_results);
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensor_ref<T>::value, T>::type call(
      T results) {
    PT_LAZY_DEBUG("Lazy Call Tuple_Of_Tensor_ref :: ", m_symbol.toQualString());
    bool isView = false;
    habana_lazy::ir::setCurrentModuleName(module_name);
    if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2) ||
        !GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING)) {
      isView = viewUpdateInputs();
    }
    std::shared_ptr<HbLazyFrontEndInfoToBackend> infoToBackEnd =
        std::make_shared<HbLazyFrontEndInfoToBackend>();
    infoToBackEnd->set_lazy_op_name(m_symbol.toQualString());

    auto context = get_device_lazy_execution_context();

    if (is_optimized_lazy_eager_supported(
            isView, context->viewContext.isLazyViewPresent)) {
      size_t lazy_eager_key = 0;
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);

      infoToBackEnd->set_optimized_lazy_eager_key(lazy_eager_key);
      infoToBackEnd->set_is_optimized_lazy_eager(IsOptimizedLazyEagerCached);
    }

    return HandleLazy(results, infoToBackEnd);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_arithmetic<T>::value, T>::type call() {
    habana_lazy::ir::setCurrentModuleName(module_name);
    viewUpdateInputs();
    const auto& node = create_node();
    const auto& t = get_inputs().at(m_out_index).toTensor();
    const auto& result =
        empty_hpu_lazy(1, t.options(), t.suggest_memory_format(), false);
    auto hl_result = GetHbLazyTensor(result, true, !m_collective_op);
    hl_result.IrSetNode(node);

    log_dev_mem_stats("Post-Accumulation", m_symbol.toQualString());
    runSBS(result);
    return result.item().template to<T>();
  }

 private:
  template <typename T = ReturnType, class U>
  typename std::enable_if<std::is_void<T>::value, T>::type call_internal_lists(
      U list) {
    habana_lazy::ir::setCurrentModuleName(module_name);

    if (!lazyEagerOptimizedViewHandling()) {
      viewUpdateInputs();
    }

    const auto& node = create_node();

    auto context = get_device_lazy_execution_context();
    int64_t out_index = 0;
    common::ListOfListsCustomIterator<U> customIt(list);
    if (!customIt.empty()) {
      do {
        auto tensors = customIt.get_next_item();
        for (const auto& tensor : tensors) {
          HbLazyTensorViews::CustomKernelAddNodeInplace(
              tensor, node, out_index);
          auto hl_result = GetHbLazyTensor(tensor, true, !m_collective_op);
          updateDstDependencies(tensor);
          context->MarkTensorStatus(
              hl_result.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
        }
        runSBS(tensors);
      } while (customIt.has_more_items());
    }

    log_dev_mem_stats("Post-Accumulation", m_symbol.toQualString());
    flush_op(out_index);
  }

 public:
  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList tensors) {
    return call_internal_lists<T>(tensors);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      const std::vector<at::Tensor>& tensors) {
    return call<T>(at::TensorList{tensors});
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      c10::ArrayRef<at::TensorList> tensorlists) {
    return call_internal_lists<T>(tensorlists);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      const std::vector<at::TensorList>& tensorlists) {
    return call<T>(c10::ArrayRef<at::TensorList>{tensorlists});
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, std::vector<at::Tensor>>::value, T>::
      type
      call() {
    habana_lazy::ir::setCurrentModuleName(module_name);
    const auto& tensors = get_result();
    const auto& node = create_node();
    int i = 0;

    for (const auto& tensor : tensors) {
      auto hl_result = GetHbLazyTensor(tensor, true, !m_collective_op);
      hl_result.IrSetNode(node, i++);
    }

    log_dev_mem_stats("Post-Accumulation", m_symbol.toQualString());
    runSBS(tensors);
    flush_op(tensors.size());

    return tensors;
  }

  template <typename T = ReturnType>
  typename std::
      enable_if<std::is_same<T, std::vector<at::Tensor>>::value, void>::type
      call(const std::vector<at::Tensor>& tensors) {
    habana_lazy::ir::setCurrentModuleName(module_name);
    const auto& node = create_node();
    int i = 0;

    for (const auto& tensor : tensors) {
      auto hl_result = GetHbLazyTensor(tensor, true, !m_collective_op);
      hl_result.IrSetNode(node, i++);
    }
    runSBS(tensors);
    flush_op(tensors.size());
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  HandleLazy(
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend =
          nullptr) {
    bool isOptimizedLazyEager = false;
    if (info_to_lazy_backend) {
      isOptimizedLazyEager =
          info_to_lazy_backend->get_is_optimized_lazy_eager();
    }

    const auto& result = get_result();
    auto hl_result = GetHbLazyTensor(result, true, !m_collective_op);
    if (isOptimizedLazyEager == false) {
      if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) &&
          info_to_lazy_backend) {
        // lazy eager - preparing the input tensor uids
        prepare_lazy_eager_input_uids(info_to_lazy_backend);

        handle_strided_inputs();
      }
      const auto& node = create_node();
      hl_result.IrSetNode(node);
    } else {
      PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
      std::vector<ir::Value> input_vals = prepare_lazy_eager_input_values();
      info_to_lazy_backend->set_input_values(std::move(input_vals));
    }

    log_dev_mem_stats("Post-Accumulation", m_symbol.toQualString());
    runSBS(result);
    flush_op(1, info_to_lazy_backend, {hl_result});
    return result;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  HandleLazy(
      at::Tensor& self,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend =
          nullptr) {
    bool isOptimizedLazyEager = false;
    if (info_to_lazy_backend) {
      isOptimizedLazyEager =
          info_to_lazy_backend->get_is_optimized_lazy_eager();
    }

    auto hl_result = GetHbLazyTensor(self, true, !m_collective_op);
    if (isOptimizedLazyEager == false) {
      // lazy eager - preparing the input tensor uids
      if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) &&
          info_to_lazy_backend) {
        prepare_lazy_eager_input_uids(info_to_lazy_backend);
      }

      const auto& node = create_node();
      hl_result.IrSetNode(node);
    } else {
      PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
      std::vector<ir::Value> input_vals = prepare_lazy_eager_input_values();
      info_to_lazy_backend->set_input_values(std::move(input_vals));
    }

    runSBS(self);
    flush_op(1, info_to_lazy_backend, {hl_result});
    // walkaround here due to the second input of rrelu will be write
    using namespace std::literals;
    const auto node_str = std::string_view{m_symbol.toQualString()};
    if (node_str == "hpu::rrelu_with_noise"sv) {
      HbLazyTensor::StepMarker({});
    }
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
    PT_LAZY_DEBUG("Lazy Call :: ", m_symbol.toQualString());
    habana_lazy::ir::setCurrentModuleName(module_name);
    bool isView = false;
    if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2) ||
        !GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING)) {
      isView = viewUpdateInputs();
    }
    std::shared_ptr<HbLazyFrontEndInfoToBackend> infoToBackEnd =
        std::make_shared<HbLazyFrontEndInfoToBackend>();
    infoToBackEnd->set_lazy_op_name(m_symbol.toQualString());
    auto context = get_device_lazy_execution_context();

    if (is_optimized_lazy_eager_supported(
            isView, context->viewContext.isLazyViewPresent)) {
      size_t lazy_eager_key = 0;
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);

      infoToBackEnd->set_optimized_lazy_eager_key(lazy_eager_key);
      infoToBackEnd->set_is_optimized_lazy_eager(IsOptimizedLazyEagerCached);
    }

    context->viewContext.isLazyViewPresent = false;

    auto result = HandleLazy(infoToBackEnd);
    if (not m_hw_scaling_ids.empty()) {
      const auto& input = m_inputs[m_hw_scaling_ids[0]];
      HABANA_ASSERT(input.isTensor());
      habana_helpers::set_output_hw_scaling_meta(input.toTensor(), result);
    }
    return result;
  }

  bool viewUpdateInputsProcessSingleTensor(at::Tensor& t, size_t& idx) {
    bool is_view = false;
    if (t.defined() && (t.device().type() == c10::DeviceType::HPU)) {
      auto hl_t = GetHbLazyTensor(t, true, !m_collective_op);

      // if it is base tensor, use the most recent version else check if
      // it is a view
      auto& base_tensor_opt = hl_t.getDataPtr()->recent_base;
      if (base_tensor_opt.has_value()) {
        // accumulation thread cannot release tensors as it can cause a
        // deadlock with GIL
        if (AccThread::Get().inAccThreadContext()) {
          auto old_tensor = std::move(m_inputs[idx]);
          AccThread::Get().PushCleanupTask(
              [old_tensor = std::move(old_tensor)]() {});
        }

        m_inputs[idx] = base_tensor_opt.value();
      } else {
        if (HbLazyTensorViews::HandleViews(t, hl_t)) {
          is_view = true;
        }
      }
    } // if (t.defined() && (

    return is_view;
  }

  bool viewUpdateInputs() {
    size_t idx = 0;
    bool is_view = false;
    for (auto& ival : m_inputs) {
      if (ival.isTensor()) {
        auto& t = ival.toTensor();
        is_view = viewUpdateInputsProcessSingleTensor(t, idx);
      } else if (ival.isTensorList()) {
        auto tl = ival.toTensorVector();
        std::vector<at::Tensor> updated_t_list;
        for (size_t i = 0; i < tl.size(); ++i) {
          auto& t = tl[i];
          if (viewUpdateInputsProcessSingleTensor(t, idx)) {
            is_view = true;
            updated_t_list.push_back(t);
          } else {
            auto t_updated = HbLazyTensorViews::get_recent_base_tensor(t);
            updated_t_list.push_back(t_updated);
          }
        } // for( size_t
        ival = updated_t_list;
      }

      idx++;
    }
    return is_view;
  }

  void HandleViewsInplace(
      const at::Tensor& self,
      habana_lazy::HbLazyTensor& hl_self,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend =
          nullptr) {
    bool isOptimizedLazyEager = false;
    if (info_to_lazy_backend) {
      if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING)) {
        isOptimizedLazyEager =
            info_to_lazy_backend->get_is_optimized_lazy_eager();
      }
    }

    auto orig_size = self.sizes();
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING)) {
      auto hb_lazy_self = GetHbLazyTensor(self);
      auto impl = hb_lazy_self.getAttachedTensorImpl();
      if (impl) {
        orig_size = impl->sizes();
      } else {
        HABANA_ASSERT("Input tensor doesn't have storage attached!");
      }
    }
    auto out_t = empty_hpu_lazy(
        orig_size, self.options(), self.suggest_memory_format(), false);

    // To Do - To check the handling for the lazy eager
    if (!is_inplace(m_symbol)) {
      // out variant needs storage as it is a graph input
      out_t = empty_hpu_lazy(
          self.sizes(), self.options(), self.suggest_memory_format(), true);

      /* b = a.view().fill_()
      This is lowered as follows:
      new_input_tensor.fill_()
      updated_a = strided_insert(a, new_tensor)
      Now, new_input_tensor needs to be added to Tmap.
    TODO SW-114041: In general HandleViewsInplace() is expected to be run only
    for cache misses.  How do we capture this new input in case of cache hits?*/
      RUNNING_HASH_COMBINE_TENSOR(out_t);

      for (int idx = (int)m_inputs.size() - 1; idx >= 0; idx--) {
        auto t = m_inputs[idx];
        if (t.isTensor() && t.toTensor().is_same(self)) {
          // accumulation thread cannot release tensors as it can cause a
          // deadlock with GIL
          if (AccThread::Get().inAccThreadContext()) {
            auto old_tensor = std::move(m_inputs[idx]);
            AccThread::Get().PushCleanupTask(
                [old_tensor = std::move(old_tensor)]() {});
          }

          m_inputs[idx] = out_t;
          // break after first update because we can cases like torch.ge(a, b,
          // out = a). In this case need to replace only out = a case
          break;
        }
      }
    }

    if (isOptimizedLazyEager == false) {
      PT_LAZY_DEBUG("Normal Lazy Eager Inplace (with Views) Path Chosen");

      if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) &&
          info_to_lazy_backend) {
        // lazy eager - preparing the input tensor uids
        prepare_lazy_eager_input_uids(info_to_lazy_backend);

        handle_strided_inputs();
      }

      const auto& node = create_node();
      hl_self = GetHbLazyTensor(out_t);

      if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING)) {
        handle_inplace_strided_output(self, hl_self, node);
      } else {
        hl_self.IrSetNode(node);

        flush_op(1);
        // add strided insert node and update most recent version of original
        // tensor
        strided_insert_hpu_lazy(self, out_t);
      }
    } else {
      PT_LAZY_DEBUG("Optimized Lazy Eager Inplace (with Views) Path Chosen");
      std::vector<ir::Value> input_vals = prepare_lazy_eager_input_values();
      info_to_lazy_backend->set_input_values(std::move(input_vals));
    }
  }

  // For inplace/out variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type
  HandleLazy(
      at::Tensor& self,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend =
          nullptr) {
    bool isOptimizedLazyEager = false;
    if (info_to_lazy_backend) {
      isOptimizedLazyEager =
          info_to_lazy_backend->get_is_optimized_lazy_eager();
    }
    auto context = get_device_lazy_execution_context();
    auto hl_self = GetHbLazyTensor(self, true, !m_collective_op);

    bool is_self_view = false;

    if (lazyEagerOptimizedViewHandling()) {
      // Checking if any of the inputs is a strided tensor
      for (size_t idx = 0; idx < m_inputs.size(); idx++) {
        auto& t = m_inputs[idx];
        if (t.isTensor()) {
          auto hl_t = GetHbLazyTensor(t.toTensor());
          if (hl_t.GetIsStrided()) {
            is_self_view = hl_t.GetIsStrided();
            break;
          }
        }
      }
    } else {
      auto& params_opt = hl_self.getDataPtr()->stride_params;
      if (params_opt.has_value()) {
        if (params_opt.value().viewStatus != kEvaluated) {
          is_self_view = true;
        }
      }
    }

    std::vector<at::IValue> sbs_stack;
    // special handling for self tensor
    if (is_self_view == false) {
      at::Tensor self_updated;
      if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING)) {
        self_updated = self;
      } else {
        // use most recent version of the tensor if applicable
        self_updated = HbLazyTensorViews::get_recent_base_tensor(self);
      }
      hl_self = GetHbLazyTensor(self_updated, true, !m_collective_op);

      // identify the inplace index and replace it with updated version
      // m_inputs will be used in create_node()
      for (size_t idx = 0; idx < m_inputs.size(); idx++) {
        auto& t = m_inputs[idx];
        if (t.isTensor() && t.toTensor().is_same(self)) {
          // accumulation thread cannot release tensors as it can cause a
          // deadlock with GIL
          if (AccThread::Get().inAccThreadContext()) {
            AccThread::Get().PushCleanupTask([t]() {});
          }

          t = self_updated;
        }
      }
      // special handling for self tensor
      // skip ctrl edges for inplace
      // TODO do the same for out variants
      if (!is_inplace(m_symbol)) {
        updateDstDependencies(self_updated);
      }
      if (isOptimizedLazyEager == false) {
        PT_LAZY_DEBUG("Normal Lazy Eager Inplace Path Chosen");

        // lazy eager - preparing the input tensor uids
        if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
            GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) &&
            info_to_lazy_backend) {
          prepare_lazy_eager_input_uids(info_to_lazy_backend);
        }

        const auto& node = create_node();
        hl_self.IrSetNode(node);
        // Special handling for SBS in inplace, before the inplace op will
        // override the tensor
        if (is_inplace(m_symbol)) {
          m_sbs_runner->populateInputForCPUOp(
              get_inputs(), node->GetMetaData(), sbs_stack);
        }
      } else {
        PT_LAZY_DEBUG("Optimized Lazy Eager Inplace Path Chosen");
        std::vector<ir::Value> input_vals = prepare_lazy_eager_input_values();
        info_to_lazy_backend->set_input_values(std::move(input_vals));
      }
    } else {
      HandleViewsInplace(self, hl_self, info_to_lazy_backend);
    }

    // numel == 0 is the correct check, need the size check until pytorch
    // fixes it properly
    // https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
    std::vector<int64_t> out_shape;
    if (m_output_meta_fn) {
      out_shape = m_output_meta_fn(get_inputs())[0].shape;
    } else if (m_out_shapes.empty())
      out_shape = get_inputs().at(m_out_index).toTensor().sizes().vec();
    else {
      out_shape = m_out_shapes[0];
    }
    if (self.sizes() != out_shape || m_shape_was_changed) {
      auto impl = hl_self.getAttachedTensorImpl();
      THHTensor_resizeNd(impl, out_shape.size(), out_shape.data(), nullptr);
      self.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
      if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
          GET_ENV_FLAG_NEW(PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH) &&
          info_to_lazy_backend) {
        info_to_lazy_backend->set_out_shapes({out_shape});
      }
    }

    context->MarkTensorStatus(
        hl_self.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);

    log_dev_mem_stats("Post-Accumulation", m_symbol.toQualString());
    runSBS(self, sbs_stack);
    flush_op(1, info_to_lazy_backend, {hl_self});
    return self;
  }

  // For inplace/out variants and regular variants with accumulation thread
  template <typename T = ReturnType>
  typename std::enable_if<
      (std::is_same<T, at::Tensor&>::value ||
       std::is_same<T, at::Tensor>::value),
      T>::type
  call(at::Tensor& self) {
    std::string node_str = m_symbol.toQualString();
    if ((node_str == "aten::mul") && (m_inputs.size() == 3)) {
      // WA to enable autogen for mul_out in lazy
      auto& params_opt =
          GetHbLazyTensor(m_inputs[2].toTensor()).getDataPtr()->stride_params;

      if (params_opt.has_value()) {
        return mul_out_hpu_lazy(
            m_inputs[0].toTensor(),
            m_inputs[1].toTensor(),
            m_inputs[2].toTensor());
      }
    }

    habana_lazy::ir::setCurrentModuleName(module_name);
    PT_LAZY_DEBUG(
        "Lazy Call Inplace/out or regular with acc thread:self :: ",
        m_symbol.toQualString());
    bool isView = false;

    if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2) ||
        !GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING)) {
      // Handle views or fetch updated tensor for all the inputs
      isView = viewUpdateInputs();
    }

    std::shared_ptr<HbLazyFrontEndInfoToBackend> infoToBackEnd =
        std::make_shared<HbLazyFrontEndInfoToBackend>();
    infoToBackEnd->set_lazy_op_name(node_str);

    auto context = get_device_lazy_execution_context();

    if (is_optimized_lazy_eager_supported(
            isView, context->viewContext.isLazyViewPresent)) {
      size_t lazy_eager_key = 0;
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);

      infoToBackEnd->set_optimized_lazy_eager_key(lazy_eager_key);
      infoToBackEnd->set_is_optimized_lazy_eager(IsOptimizedLazyEagerCached);
    }

    context->viewContext.isLazyViewPresent = false;

    return HandleLazy(self, infoToBackEnd);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, T>::type
  HandleLazy(
      const at::Tensor& self,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend =
          nullptr) {
    std::vector<int64_t> out_shape;
    if (m_output_meta_fn) {
      out_shape = m_output_meta_fn(get_inputs())[0].shape;
    } else if (m_out_shapes.empty())
      out_shape = get_inputs().at(m_out_index).toTensor().sizes().vec();
    else {
      out_shape = m_out_shapes[0];
    }

    const bool need_resize = self.sizes() != out_shape || m_shape_was_changed;

    if (need_resize) {
      // SyncAccThreadPool to before resize an intermediate tensor
      if (!self.storage().data_ptr())
        HbLazyTensor::StepMarker({});
    }

    auto hl_self = GetHbLazyTensor(self, true, !m_collective_op);

    if (need_resize) {
      // Handle View Input for TensorResize
      if (!hl_self.isStorageAttached()) {
        auto& stride_params_opt = hl_self.getDataPtr()->stride_params;
        if (stride_params_opt.has_value()) {
          if (self.device().type() == c10::DeviceType::HPU) {
            flush_op();
            // Trigger point execution
            PT_IRGRAPH_DEBUG("step marker due to view tensor resize");
            HbLazyTensor::StepMarker({});
          }
          auto base = HbLazyTensorViews::get_recent_base_tensor(
              stride_params_opt.value().base);
          TORCH_CHECK(base.storage(), "base tensor should have valid storage");
          auto base_internal_tensor = GetHbLazyTensor(base).CurrentTensorData();
          hl_self.SetTensorData(*base_internal_tensor);
        } else {
          TORCH_CHECK(
              0, "Neither storage attached to input tensor, not its view.")
        }
      }
      auto impl = hl_self.getAttachedTensorImpl();
      THHTensor_resizeNd(impl, out_shape.size(), out_shape.data(), nullptr);
      hl_self.ClearStrideParams();
      self.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
      if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
          GET_ENV_FLAG_NEW(PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH) &&
          info_to_lazy_backend) {
        info_to_lazy_backend->set_out_shapes({out_shape});
      }
    }

    updateDstDependencies(self);
    const auto& node = create_node();
    hl_self.IrSetNode(node);

    auto context = get_device_lazy_execution_context();
    context->MarkTensorStatus(
        hl_self.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);

    log_dev_mem_stats("Post-Accumulation", m_symbol.toQualString());
    runSBS(self);
    flush_op(1, std::move(info_to_lazy_backend));
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, T>::type
  call(const at::Tensor& self) {
    PT_LAZY_DEBUG("Lazy Call Inplace :: ", m_symbol.toQualString());
    habana_lazy::ir::setCurrentModuleName(module_name);
    std::shared_ptr<HbLazyFrontEndInfoToBackend> infoToBackEnd =
        std::make_shared<HbLazyFrontEndInfoToBackend>();
    infoToBackEnd->set_lazy_op_name(m_symbol.toQualString());
    // Temporarily disabled the switch - To Do
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_OPTIM_CACHE) && false) {
      size_t lazy_eager_key = 0;
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);

      infoToBackEnd->set_optimized_lazy_eager_key(lazy_eager_key);
      infoToBackEnd->set_is_optimized_lazy_eager(IsOptimizedLazyEagerCached);
    }

    return HandleLazy(self, infoToBackEnd);
  }

  // helper function to inspect inplace/out tensors handled by LazyOp
  void inspect_result(const at::Tensor& tensor) {
    /* Same check happens in GetHbLazyTensor, but it's in acc thread.*/
    /* Make sure in main thread, that we get HPU tensor .*/
    HABANA_ASSERT(
        tensor.device().type() == at::kHPU,
        "Got a non-HPU tensor, expecting an HPU tensor");

    // In case of _out ops or resize_, the output tensor may come with wrong or
    // empty shape. There is mechanism to handle it at HandleLazy level, but we
    // need to set the correct shape on at::Tensor so it's propagated to Python
    // in main thread.
    std::vector<int64_t> out_shape;
    if (m_output_meta_fn) {
      out_shape = m_output_meta_fn(get_inputs())[0].shape;
    } else if (m_out_shapes.empty()) {
      out_shape = get_inputs().at(m_out_index).toTensor().sizes().vec();
    } else {
      out_shape = m_out_shapes[0];
    }
    if (tensor.sizes() != out_shape) {
      using namespace std::literals;
      // only raise warning for inplace/out case
      if (std::string_view(m_symbol.toQualString()) != "aten::resize_"sv &&
          tensor.numel() != 0) {
        TORCH_WARN(
            "An output with one or more elements was resized since it had ",
            "shape ",
            tensor.sizes(),
            ", which does not match the required ",
            "output shape ",
            c10::ArrayRef<int64_t>(out_shape),
            ". ",
            "This behavior is deprecated, and in a future PyTorch release outputs ",
            "will not be resized unless they have zero elements. You can explicitly ",
            "reuse an out tensor t by resizing it, inplace, to zero elements with ",
            "t.resize_(0).");
      }
      PT_IRGRAPH_DEBUG("step marker due to out shape changed");
      HbLazyTensor::StepMarker({});
      tensor.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
      set_shape_changed();
    }
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type
  get_result(at::Tensor& tensor) {
    inspect_result(tensor);
    return tensor;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, void>::type
  get_result(const at::Tensor& tensor) {
    inspect_result(tensor);
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensor_ref<T>::value, T>::type get_result(
      T tensors) {
    habana::OutputMetaDataVector meta;
    if (m_output_meta_fn) {
      meta = m_output_meta_fn(get_inputs());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          std::tuple_size<T>::value == meta.size());
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          std::tuple_size<T>::value == m_out_shapes.size());
    }

    int i = 0;
    habana::for_each_in_tuple(tensors, [&, this](auto& tensor) {
      /* Same check happens in GetHbLazyTensor, but it's in acc thread.*/
      /* Make sure in main thread, that we get HPU tensor .*/
      HABANA_ASSERT(
          tensor.device().type() == at::kHPU,
          "Got a non-HPU tensor, expecting an HPU tensor");
      // In case of _out ops, the output tensor may come with wrong or empty
      // shape. There is mechanism to handle it at HandleLazy level, but we
      // need to set the correct shape on at::Tensor so it's propagated to
      // Python in main thread.
      std::vector<int64_t> out_shape;
      if (meta.size()) {
        out_shape = meta[i].shape;
      } else if (m_out_shapes.empty()) {
        out_shape = tensor.sizes().vec();
      } else {
        out_shape = m_out_shapes[i];
      }

      if (tensor.sizes() != out_shape) {
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
        m_shape_was_changed_in_tuple.push_back(true);
      } else {
        m_shape_was_changed_in_tuple.push_back(false);
      }
      i++;
    });

    return tensors;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  get_result() {
    PT_LAZY_TRACE;
    if (m_output_meta_fn) {
      auto meta = m_output_meta_fn(get_inputs());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(meta.size() == 1);
      auto output_meta = meta[0];
      return empty_hpu_lazy(
          output_meta.shape, output_meta.dtype, output_meta.mem_format, false);
    }

    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }

    if (m_out_meta_tensors.empty()) {
      const auto& t = get_inputs().at(m_out_index).toTensor();
      const auto& out_shape =
          m_out_shapes.empty() ? t.sizes() : m_out_shapes[0];
      auto options = t.options();
      if (m_scalar_types.size()) {
        HABANA_ASSERT(m_scalar_types.size() == 1);
        options = options.dtype(m_scalar_types[0]);
      }
      return empty_hpu_lazy(
          out_shape, options, t.suggest_memory_format(), false);
    }

    const auto& t = m_out_meta_tensors[0];
    return empty_hpu_lazy(
        t.sizes(), t.options(), t.suggest_memory_format(), false);
  }

  template <typename T = ReturnType>
  typename std::enable_if<not is_tuple_of_tensor_ref<T>::value, T>::type
  get_result() {
    PT_LAZY_TRACE;
    if (m_output_meta_fn) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_out_index == 0);
      const auto& meta = m_output_meta_fn(get_inputs());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          std::tuple_size<T>::value == meta.size());
      unsigned i = 0;
      ReturnType results;

      habana::for_each_in_tuple(results, [&](auto& result) {
        auto output_meta = meta[i++];
        result = empty_hpu_lazy(
            output_meta.shape,
            output_meta.dtype,
            output_meta.mem_format,
            false);
      });
      return results;
    }

    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        std::tuple_size<T>::value == m_out_shapes.size());

    unsigned i = 0;
    ReturnType results;

    habana::for_each_in_tuple(results, [&](auto& result) {
      auto t = get_inputs().at(m_out_index).toTensor();
      auto options = t.options();
      if (m_scalar_types.size()) {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(i < m_scalar_types.size());
        options = options.dtype(m_scalar_types[i]);
      }
      result = empty_hpu_lazy(
          m_out_shapes[i++], options, t.suggest_memory_format(), false);
    });
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, std::vector<at::Tensor>>::value, T>::
      type
      get_result() {
    PT_LAZY_TRACE;
    if (m_output_meta_fn) {
      const auto& meta = m_output_meta_fn(get_inputs());
      std::vector<at::Tensor> results;
      results.reserve(meta.size());

      for (auto output_meta : meta) {
        results.emplace_back(empty_hpu_lazy(
            output_meta.shape,
            output_meta.dtype,
            output_meta.mem_format,
            false));
      }
      return results;
    }
    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }
    const auto results_size = m_out_shapes.size();
    HABANA_ASSERT(results_size != 0);

    std::vector<at::Tensor> results;
    results.reserve(results_size);
    auto options = at::TensorOptions(at::kHPU);
    if (m_scalar_types.empty()) {
      auto dtype = get_inputs().at(m_out_index).toTensor().scalar_type();
      for (const auto& out_shape : m_out_shapes) {
        results.emplace_back(at::empty(
            out_shape, options.dtype(dtype), at::MemoryFormat::Contiguous));
      }
    } else {
      HABANA_ASSERT(m_scalar_types.size() == results_size);
      for (size_t i = 0; i < results_size; ++i) {
        results.emplace_back(at::empty(
            m_out_shapes[i],
            options.dtype(m_scalar_types[i]),
            at::MemoryFormat::Contiguous));
      }
    }
    return results;
  }

  const std::vector<std::vector<int64_t>>& get_out_shapes() const {
    return m_out_shapes;
  }

  // Helper function to mark, that output shape of at::Tensor has been changed.
  // Needed for _out ops to make sure we add potential resize op.
  void set_shape_changed() {
    m_shape_was_changed = true;
  }

  const std::vector<at::IValue>& inputs() const {
    return m_inputs;
  }

  const c10::Symbol& symbol() const {
    return m_symbol;
  }

  void set_scalar_types(const std::vector<c10::ScalarType>& scalar_types) {
    m_scalar_types = scalar_types;
  }

  [[nodiscard]] const std::vector<c10::ScalarType>& get_scalar_types() const {
    return m_scalar_types;
  }

  void set_hw_scaling_index(const std::vector<int> ids) {
    m_hw_scaling_ids = ids;
  }

  void SetSTMetaFn(std::function<bool(
                       habana_helpers::IShapeList& inputs,
                       habana_helpers::IShapeList& outputs)> fn) {
    static_cast<void>(fn);
  }

  void SetOutputMetaFn(
      std::function<habana::OutputMetaDataVector(const at::Stack&)>
          output_meta) {
    m_output_meta_fn = std::move(output_meta);
  }

 private:
  bool isMetadataCandidate(const at::IValue& input) const {
    return input.isBool() || input.isDevice() || input.isIntList() ||
        input.isDoubleList() || input.isBoolList() || input.isString() ||
        input.isNone() ||
        (input.isList() &&
         !input.toList().elementType()->cast<at::TensorType>());
  }

  template <typename ValuesT = ir::ValueList>
  void create_inputs(
      ValuesT& values,
      std::vector<at::Tensor>& input_pt_vec,
      ir::MetaData& metadata,
      bool is_optimized_lazy_eager) {
    auto context = get_device_lazy_execution_context();
    values.reserve(m_inputs.size());
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      const at::IValue& input = m_inputs[i];
      if (input.isScalar()) {
        // Already taken care in optimized lazy eager JIT graph key
        // calculation so not required for the optimized lazy eager
        if (!is_optimized_lazy_eager) {
          auto val = GetIrValueForScalar(input.toScalar());
          values.emplace_back(val);
        } else {
          continue;
        }
      } else if (isMetadataCandidate(input)) {
        // Not supported for optimized lazy eager - To Do
        if (!is_optimized_lazy_eager) {
          metadata.set(input, i);
        } else {
          continue;
        }
      } else if (input.isTensor()) {
        const at::Tensor& t = input.toTensor();
        if (t.defined()) {
          HABANA_ASSERT(t.device().type() == c10::DeviceType::HPU)
          auto val = GetHbLazyTensor(t, true, !m_collective_op).GetIrValue();
          if (!is_optimized_lazy_eager) {
            values.emplace_back(val);
            input_pt_vec.emplace_back(t);
          } else {
            // Taking care of duplicate values here itself for optimized lazy
            // eager. In normal flow it is taken care later in the flow. To Do
            // - To make it same for normal flow as well.
            auto it = std::find(values.begin(), values.end(), val);
            if (it == values.end()) {
              values.emplace_back(val);
            }
            if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT)) {
              context->m_retained_tensor_list.emplace_back(t);
            }
          }
        } else {
          if (!is_optimized_lazy_eager) {
            // Already taken care in optimized lazy eager JIT graph key
            // calculation
            metadata.set(torch::jit::IValue(), i);
          }
        }
      } else if (input.isList()) {
        const auto& list = input.toListRef();
        ir::ValueList opt_tensors;
        std::vector<at::Tensor> list_input_pt_vec;
        bool is_optional = false;

        for (const auto& li : list) {
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
              li.isNone() or li.isTensor(),
              "Got unhandled list item type: ",
              li.tagKind(),
              " for ",
              m_symbol.toQualString(),
              " at index ",
              i,
              ".");
          if (li.isNone()) {
            if (!is_optimized_lazy_eager) {
              // Not supported/required for optimized lazy eager - To Do
              opt_tensors.emplace_back(GetIrValueForNone());
              is_optional |= true;
            }
          } else {
            const auto& t = li.toTensor();
            if (!is_optimized_lazy_eager) {
              opt_tensors.emplace_back(
                  GetHbLazyTensor(t, true, !m_collective_op).GetIrValue());
              list_input_pt_vec.emplace_back(t);
            } else {
              // Taking care of duplicate values here itself for optimized
              // lazy eager. In normal flow it is taken care later in the
              // flow. To Do
              // - To make it same for normal flow as well.
              auto val =
                  GetHbLazyTensor(t, true, !m_collective_op).GetIrValue();
              auto it = std::find(values.begin(), values.end(), val);
              if (it == values.end()) {
                values.emplace_back(val);
              }
              if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT)) {
                context->m_retained_tensor_list.emplace_back(t);
              }
            }
          }
        }

        // Not required in optimized lazy eager flow as values are already
        // prepared on per tensor basis.
        if (!is_optimized_lazy_eager) {
          const auto& list_input =
              GetIrValueForListConstruct(opt_tensors, is_optional);
          list_input.mp_node->AddInputPtTensors(list_input_pt_vec);
          values.emplace_back(list_input);
        }
      } else {
        PT_BRIDGE_FATAL(
            "Got unhandled type: ",
            input.tagKind(),
            " for ",
            m_symbol.toQualString(),
            " at index ",
            i);
        HABANA_ASSERT(0);
      }
    }
  }

 protected:
  std::vector<at::IValue>& get_inputs() {
    return m_inputs;
  }

  std::string get_module_name() {
    return module_name;
  }

  void set_broadcast_details(std::vector<bool>&& bcast_vec) {
    m_bcast_details = std::move(bcast_vec);
  }

  void set_inputs(const std::vector<at::IValue>& inputs) {
    auto inputsHpu = inputs;
    for (auto& t : inputsHpu) { // Any tensor on CPU needs to be moved to HPU
      if (t.isTensor() && t.toTensor().defined()) {
        const at::Tensor& tensor = t.toTensor();
        if (t.toTensor().device().type() != c10::DeviceType::HPU) {
          at::Tensor tinput;
          if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
            // is_wrapped_number: True if a tensor was auto-wrapped from a
            // C++ or Python number.
            if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
              // For lazy eager Skip scalar handling at FE, Use scalar Ivalue as
              // input
              t = c10::IValue(tensor.item());
            } else {
              // If the CPU tensor is a wrapped number, then use
              // get_tensor_for_scalar method to retrieve cached HPU tensors for
              // the scalar value
              auto dtype = tensor.scalar_type();
              tinput = get_tensor_for_scalar(
                  tensor.item().toDouble(), at::TensorOptions().dtype(dtype));
              t = c10::IValue(tinput);
            }
          } else {
            // Use non_blocking .to()
            tinput = tensor.to(c10::kHPU, true);
            t = c10::IValue(tinput);
          }
        } else {
          // hpu input tensors
          GetHbLazyTensor(tensor, true, false).SetOpAccumulationInProgress();
        }
      }

      if (!m_collective_op)
        handle_collective(t);
    }
    m_sbs_runner->setCPUInputs(inputsHpu);
    m_inputs = inputsHpu;
    // [toDo] this is for gtest we will eventually move to MACRO
    RUNNING_HASH_COMBINE_OPERATOR_STR(m_symbol, m_inputs);
  }

  virtual ReturnType get_result_overrideable() {
    HABANA_ASSERT(
        0,
        "out_index is negative, implement get_result_overrideable() in your op.");
    // Call std::terminate here to avoid compilation error due to no return
    // statement. return cannot be here because sometimes the type is Tensor&
    // or a tuple of tensors. This terminate is never reachable though.
    std::terminate();
  }

  inline bool is_optimized_lazy_eager_supported(
      bool is_view,
      bool is_lazy_view_present) {
    return (
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_OPTIM_CACHE) && !is_view &&
        !is_lazy_view_present);
  }

  // JIT IR Cache key calculation for optimized lazy eager
  size_t calculate_optimized_lazy_eager_key() {
    size_t optimized_key = static_cast<uint32_t>(m_symbol);
    optimized_key = at::hash_combine(optimized_key, m_out_shapes.size());

    optimized_key = at::hash_combine(
        optimized_key,
        habana::HPURegistrar::get_hpu_global_config().getDeterministic() ||
            at::globalContext().deterministicAlgorithms());

    std::unordered_set<size_t> input_hash_values;
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      optimized_key = at::hash_combine(optimized_key, i);
      const at::IValue& input = m_inputs[i];
      // Create stack based on input tensors / tensor lists.
      // Metadata and scalars are part of key calculation, so we skip them.
      if (input.isScalar()) {
        optimized_key =
            at::hash_combine(optimized_key, at::IValue::hash(input.toScalar()));
        continue;
      }

      if (input.isTensor()) {
        const at::Tensor& t = input.toTensor();
        if (t.defined()) {
          if (t.device().type() != c10::DeviceType::HPU) {
            // non HPU tensors to be handled later
            optimized_key = 0;
            break;
          }
          // Calculate hash based on unique tensor inputs.
          size_t input_hash_val = at::IValue::hash(input);
          if (input_hash_values.count(input_hash_val)) {
            continue;
          }
          input_hash_values.emplace(input_hash_val);
          update_hash_key_for_tensor(t, optimized_key);
          if (optimized_key == 0) {
            break;
          }
        } else {
          optimized_key = at::hash_combine(
              optimized_key, at::IValue::hash(torch::jit::IValue()));
        }
      } else if (input.isTensorList()) {
        // Not handled so returning null key
        optimized_key = 0;
        break;
      } else if (isMetadataCandidate(input)) {
        if (input.isList()) {
          for (auto& v : input.toListRef()) {
            optimized_key = ir::MetaData::ival_hash(v, optimized_key);
          }
        } else {
          optimized_key = ir::MetaData::ival_hash(input, optimized_key);
        }
      }
    }

    return optimized_key;
  }

  bool calculate_key_and_check_optimized_lazy_eager_cache(
      size_t& lazy_eager_key) {
    bool IsCached = false;
    if (!(std::getenv("PT_HPU_LAZY_CACHE_DISABLE"))) {
      lazy_eager_key = calculate_optimized_lazy_eager_key();
      PT_LAZY_DEBUG("Optimized Lazy Eager Key :: ", lazy_eager_key);
      if (lazy_eager_key != 0) {
        IsCached =
            habana::OptimizedJitGraphCache::GetOptimizedJitCache().IsCached(
                lazy_eager_key);
      }
    }

    return IsCached;
  }

  std::vector<ir::Value> prepare_lazy_eager_input_values() {
    std::vector<ir::Value> values;
    std::vector<at::Tensor> input_pt_vec;
    ir::MetaData metadata;

    create_inputs(values, input_pt_vec, metadata, true);
    return values;
  }

  void prepare_lazy_eager_input_uids(
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend) {
    std::vector<ir::Value> input_vals = prepare_lazy_eager_input_values();
    std::vector<uint64_t> uids{};
    for (auto in : input_vals) {
      std::shared_ptr<Data> d = in.m_data_ptr.lock();
      uids.emplace_back(d->unique_id);
    }
    info_to_lazy_backend->set_lazy_eager_op_input_uids(std::move(uids));
  }

  void handle_strided_input(const at::Tensor& self, size_t idx) {
    auto impl = self.unsafeGetTensorImpl();
    auto size = impl->sizes();
    auto stride = impl->strides();
    auto offset = impl->storage_offset();
    auto hl_self = GetHbLazyTensor(self);

    PT_LAZY_EAGER_DEBUG(
        "[LAZY EAGER VIEW] strided input (frontend) size : ",
        size,
        " strides : ",
        stride,
        " offset : ",
        offset);

    at::Tensor op_input_tensor = empty_hpu_lazy(
        size,
        self.options(),
        c10::nullopt,
        false,
        DATA_TENSOR,
        c10::nullopt,
        false);

    auto hl_op_input_tensor = GetHbLazyTensor(op_input_tensor);

    PT_LAZY_EAGER_DEBUG(
        "[LAZY EAGER VIEW] op input (frontend) size : ",
        op_input_tensor.unsafeGetTensorImpl()->sizes(),
        " strides : ",
        op_input_tensor.unsafeGetTensorImpl()->strides(),
        " offset : ",
        op_input_tensor.unsafeGetTensorImpl()->storage_offset());

    ir::NodePtr node = nullptr;

    node = std::make_shared<habana_lazy::ir::StridedView>(
        self, size, stride, offset, "hpu::strided_view");
    hl_op_input_tensor.IrSetNode(node);
    m_inputs[idx] = op_input_tensor;
  }

  void handle_strided_inputs() {
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      const at::IValue& input = m_inputs[i];
      if (input.isTensor()) {
        const at::Tensor& t = input.toTensor();
        if (t.defined()) {
          if (GetHbLazyTensor(t).GetIsStrided()) {
            at::Tensor strided_tensor = m_inputs[i].toTensor();
            handle_strided_input(strided_tensor, i);
          }
        }
      }
    }
  }

  void handle_inplace_strided_output(
      const at::Tensor& self,
      const HbLazyTensor& hl_self,
      ir::NodePtr node) {
    auto impl = self.unsafeGetTensorImpl();
    auto size = impl->sizes();
    auto stride = impl->strides();
    auto offset = impl->storage_offset();

    at::Tensor op_output_tensor = empty_hpu_lazy(
        size,
        self.options(),
        c10::nullopt,
        false,
        DATA_TENSOR,
        c10::nullopt,
        false);
    auto hl_op_output_tensor = GetHbLazyTensor(op_output_tensor);
    hl_op_output_tensor.IrSetNode(node);

    std::string node_str = "hpu::strided_insert";

    auto strided_node = std::make_shared<ir::StridedInsert>(
        self, op_output_tensor, stride, offset, node_str);
    hl_self.IrSetNode(strided_node);
  }

  // The Side-By-Side (SBS) Debug Tool is a debug capability for comparing
  // between tensors that are calculated by HPU to tensors that are calculated
  // by CPU.
  // Run it by adding the env var PT_SBS with one of the enum values described
  // here: debug_utils.h :: SBSModes
  // See more here:
  // https://confluence.habana-labs.com/display/SYN/Side-By-Side+Debug+Tool
  void runSBS(
      const at::TensorList results,
      const std::vector<at::IValue>& preallocated_stack =
          std::vector<at::IValue>()) {
    if (GET_ENV_FLAG_NEW(PT_SBS) != SBSModes::SBS_MODE_DISABLED) {
      PT_LAZY_DEBUG("Calling runSBS for op: ", m_symbol.toQualString());
      m_sbs_runner->run(results, get_inputs(), preallocated_stack);
    }
  }

  template <typename N = NodeConstruct>
  std::enable_if_t<std::is_class<N>::value, ir::NodePtr> create_node() {
    return m_node;
  }

  template <typename N = NodeConstruct>
  std::enable_if_t<!std::is_class<N>::value, ir::NodePtr> create_node() {
    ir::InlinedValueList values;
    std::vector<at::Tensor> input_pt_vec;
    ir::MetaData metadata;

    create_inputs(values, input_pt_vec, metadata, false);
    auto node = ir::Node::Create(m_symbol, values);
    // node->SetModuleName(module_name);
    if (metadata.size()) {
      node->SetMetaData(metadata);
    }

    node->AddInputPtTensors(input_pt_vec);
    node->set_broadcast_details(std::move(m_bcast_details));
    return node;
  }

 private:
  ir::NodePtr m_node = nullptr;
  const at::Symbol m_symbol;
  std::vector<bool> m_bcast_details;
  std::vector<std::vector<int64_t>> m_out_shapes;
  const int m_out_index;
  std::vector<int> m_hw_scaling_ids;
  at::TensorList m_out_meta_tensors = {};
  std::vector<at::IValue> m_inputs = {};
  std::vector<c10::ScalarType> m_scalar_types;
  std::function<habana::OutputMetaDataVector(const at::Stack&)>
      m_output_meta_fn;
  const std::shared_ptr<SBSInterface> m_sbs_runner;
  std::string module_name = std::string();
  bool m_shape_was_changed =
      false; // bool for changed input shape for _out ops (non-tuple input)
  std::vector<bool> m_shape_was_changed_in_tuple =
      {}; // vector of bools for any changed shapes in input tuple for _out
          // ops
  // (tuple input)
  bool m_collective_op = false;
  void update_hash_key_for_tensor(const at::Tensor& t, size_t& optimized_key) {
    auto hl_tensor = TryGetHbLazyTensor(t, true, !m_collective_op);
    if (hl_tensor) {
      // To Do - To always use the front end tensor for key calculation as it
      // might be problematic to use backend internal tensor while pipelining.
      auto val = hl_tensor->GetIrValue();
      optimized_key = at::hash_combine(optimized_key, (size_t)t.dim());
      optimized_key =
          at::hash_combine(optimized_key, static_cast<size_t>(t.scalar_type()));
      optimized_key = at::hash_combine(
          optimized_key, static_cast<size_t>(t.suggest_memory_format()));
      optimized_key =
          at::hash_combine(optimized_key, (size_t)hl_tensor->GetTensorLayout());
      if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING)) {
        bool is_strided = hl_tensor->GetIsStrided();
        optimized_key = at::hash_combine(optimized_key, (size_t)is_strided);
        // sizes, strides and offset would not be the part of the key
        // calculation once get/set node params implementation is done
        if (is_strided) {
          auto impl = t.unsafeGetTensorImpl();
          auto size_vec = impl->sizes().vec();
          auto stride_vec = impl->strides().vec();
          auto offset = impl->storage_offset();
          for (size_t k = 0; k < size_vec.size(); k++) {
            optimized_key =
                at::hash_combine(optimized_key, (size_t)size_vec.at(k));
            optimized_key =
                at::hash_combine(optimized_key, (size_t)stride_vec.at(k));
            optimized_key = at::hash_combine(optimized_key, (size_t)offset);
          }
        }
      }
      if (val.mp_node && !(val.mp_node->is_input())) {
        optimized_key = 0;
      }
    }
  }
};

template <typename ReturnType>
class LazyBinaryOp : public LazyOp<ReturnType> {
 public:
  explicit LazyBinaryOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      bool is_outfn,
      bool safe_cast_check,
      const std::vector<std::vector<int64_t>>& out_shapes = {},
      int out_index = 0)
      : LazyOp<ReturnType>(qualstring, inputs, out_shapes, out_index),
        is_outfn_(is_outfn),
        safe_cast_check_(safe_cast_check) {}

  explicit LazyBinaryOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      bool is_outfn,
      bool safe_cast_check,
      const at::TensorList& output_meta_tensors)
      : LazyOp<ReturnType>(qualstring, inputs, output_meta_tensors),
        is_outfn_(is_outfn),
        safe_cast_check_(safe_cast_check) {}

  virtual ~LazyBinaryOp() = default;

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
    auto inputs = LazyOp<T>::get_inputs();

    habana_lazy::ir::setCurrentModuleName(LazyOp<T>::get_module_name());
    if (!GET_ENV_FLAG_NEW(PT_DISABLE_DTYPE_PROMOTION)) {
      c10::optional<const at::IValue*> output = is_outfn_
          ? c10::make_optional<const at::IValue*>(&inputs.back())
          : c10::nullopt;
      auto dtype_helper =
          habana_helpers::DTypeHelper::binary_op_with_type_promotion(
              inputs, output, safe_cast_check_);

      auto compute_dtype = dtype_helper.get_common_dtype(false, false);
      dst_dtype_ = dtype_helper.get_result_dtype();

      auto inputs_updated = false;
      for (size_t i = 0; i < 2; ++i) {
        if (!inputs[i].isTensor()) {
          continue;
        }
        auto tensor_promote = inputs[i].toTensor();
        if (compute_dtype == tensor_promote.scalar_type()) {
          continue;
        }

        inputs_updated = true;
        auto self = empty_hpu_lazy(
            tensor_promote.sizes(),
            tensor_promote.options().dtype(compute_dtype).device(at::kHPU),
            tensor_promote.suggest_memory_format(),
            false);
        self = copy_hpu_lazy_(self, tensor_promote, true);

        // accumulation thread cannot release tensors as it can cause a
        // deadlock with GIL
        if (AccThread::Get().inAccThreadContext()) {
          auto old_tensor = std::move(inputs[i]);
          AccThread::Get().PushCleanupTask(
              [old_tensor = std::move(old_tensor)]() {});
        }
        inputs[i] = self;
      }

      if (inputs_updated) {
        LazyOp<T>::set_inputs(inputs);
      }
    }

    PT_LAZY_DEBUG("binary op");
    if (inputs[0].isTensor() && inputs[1].isTensor())
      LazyOp<T>::set_broadcast_details(get_broadcast_details(
          inputs[0].toTensor().sizes(), inputs[1].toTensor().sizes()));

    auto results = LazyOp<T>::call();
    return results;
  }

  // For inplace binary, the promoted type takes the self's type
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type call(
      at::Tensor& self) {
    auto inputs = LazyOp<T>::get_inputs();
    habana_lazy::ir::setCurrentModuleName(LazyOp<T>::get_module_name());

    if (!GET_ENV_FLAG_NEW(PT_DISABLE_DTYPE_PROMOTION)) {
      // Perform type promotion and validate if promoted type can be casted to
      // output data type.
      auto output = c10::make_optional<const at::IValue*>(
          is_outfn_ ? &inputs.back() : &inputs.front());
      auto dtype_helper =
          habana_helpers::DTypeHelper::binary_op_with_type_promotion(
              inputs, output, safe_cast_check_);

      auto compute_dtype = dtype_helper.get_common_dtype(false, false);
      dst_dtype_ = dtype_helper.get_result_dtype();

      auto inputs_updated = false;
      for (size_t i = 0; i < 2; ++i) {
        if (!inputs[i].isTensor()) {
          continue;
        }
        auto tensor_promote = inputs[i].toTensor();
        if (compute_dtype == tensor_promote.scalar_type()) {
          continue;
        }

        inputs_updated = true;
        auto self = empty_hpu_lazy(
            tensor_promote.sizes(),
            tensor_promote.options().dtype(compute_dtype).device(at::kHPU),
            tensor_promote.suggest_memory_format(),
            false);
        self = copy_hpu_lazy_(self, tensor_promote, true);

        // accumulation thread cannot release tensors as it can cause a
        // deadlock with GIL
        if (AccThread::Get().inAccThreadContext()) {
          auto old_tensor = std::move(inputs[i]);
          AccThread::Get().PushCleanupTask(
              [old_tensor = std::move(old_tensor)]() {});
        }
        inputs[i] = self;
      }

      if (inputs_updated) {
        LazyOp<T>::set_inputs(inputs);
      }
    }

    PT_LAZY_DEBUG("binary op inplace");
    if (inputs[0].isTensor() && inputs[1].isTensor())
      LazyOp<T>::set_broadcast_details(get_broadcast_details(
          inputs[0].toTensor().sizes(), inputs[1].toTensor().sizes()));

    return LazyOp<T>::call(self);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, void>::type call(
      at::Tensor& self) {
    auto inputs = LazyOp<T>::get_inputs();
    habana_lazy::ir::setCurrentModuleName(LazyOp<T>::get_module_name());

    if (!GET_ENV_FLAG_NEW(PT_DISABLE_DTYPE_PROMOTION)) {
      // Perform type promotion and validate if promoted type can be casted to
      // output data type.
      at::IValue ivalue(self);
      auto output = c10::make_optional<const at::IValue*>(&ivalue);
      auto dtype_helper =
          habana_helpers::DTypeHelper::binary_op_with_type_promotion(
              inputs, output, safe_cast_check_);

      auto compute_dtype = dtype_helper.get_common_dtype(false, false);
      dst_dtype_ = dtype_helper.get_result_dtype();

      auto inputs_updated = false;
      for (size_t i = 0; i < 2; ++i) {
        if (!inputs[i].isTensor()) {
          continue;
        }
        auto tensor_promote = inputs[i].toTensor();
        if (compute_dtype == tensor_promote.scalar_type()) {
          continue;
        }

        inputs_updated = true;
        auto self = empty_hpu_lazy(
            tensor_promote.sizes(),
            tensor_promote.options().dtype(compute_dtype).device(at::kHPU),
            tensor_promote.suggest_memory_format(),
            false);
        self = copy_hpu_lazy_(self, tensor_promote, true);

        // accumulation thread cannot release tensors as it can cause a
        // deadlock with GIL
        if (AccThread::Get().inAccThreadContext()) {
          auto old_tensor = std::move(inputs[i]);
          AccThread::Get().PushCleanupTask(
              [old_tensor = std::move(old_tensor)]() {});
        }
        inputs[i] = self;
      }

      if (inputs_updated) {
        LazyOp<T>::set_inputs(inputs);
      }
    }

    PT_LAZY_DEBUG("binary op");
    if (inputs[0].isTensor() && inputs[1].isTensor())
      LazyOp<T>::set_broadcast_details(get_broadcast_details(
          inputs[0].toTensor().sizes(), inputs[1].toTensor().sizes()));

    LazyOp<T>::call(self);
  }

 private:
  c10::ScalarType dst_dtype_ = c10::ScalarType::Undefined;
  bool is_outfn_ = false;
  bool safe_cast_check_ = false;

  ReturnType get_result_overrideable() override;

  std::vector<bool> get_broadcast_details(
      c10::IntArrayRef a,
      c10::IntArrayRef b) {
    size_t dimsA = a.size();
    size_t dimsB = b.size();
    size_t ndim = dimsA > dimsB ? dimsA : dimsB;

    std::vector<bool> bcast_vec;
    bcast_vec.reserve(2 * ndim);
    // Use ptrdiff_t to ensure signed comparison.
    for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
      bool is_broadcast_a = false;
      bool is_broadcast_b = false;
      ptrdiff_t offset = ndim - 1 - i;
      ptrdiff_t dimA = dimsA - 1 - offset;
      ptrdiff_t dimB = dimsB - 1 - offset;
      int64_t sizeA = (dimA >= 0) ? a[dimA] : 1;
      int64_t sizeB = (dimB >= 0) ? b[dimB] : 1;
      if ((sizeA == 1) ^ (sizeB == 1)) {
        if (sizeA == 1)
          is_broadcast_a = true;
        else
          is_broadcast_b = true;
      }
      bcast_vec.push_back(is_broadcast_a);
      bcast_vec.push_back(is_broadcast_b);
    }
    PT_LAZY_DEBUG("bcast_vec : ", bcast_vec);
    return bcast_vec;
  }
};

} // namespace habana_lazy
