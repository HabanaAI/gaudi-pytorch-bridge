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

#include <tuple>
#include <utility>

#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hpu_lazy_cache.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/sbs_runner.h"
#include "habana_lazy/view_utils.h"
#include "hpu_ops/hpu_op_helper.h"
#include "lazy_kernels_declarations.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"
#include "resize.h"

namespace habana_lazy {
enum Bool : unsigned short { bFalse = 0, bTrue = 1 };
at::Tensor permute_wt_hpu(const at::Tensor& self);
void AddMemcpy(const at::Tensor& src, at::Tensor& dst);
at::Tensor append_to_batch_h2d_list(const at::Tensor& scalar_tensor);
void updateDstDependencies(
    habana_lazy::HbLazyTensor& hl_dst,
    const at::Tensor& dst,
    bool in_place = false);

at::Tensor empty_as_strided_lazy(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset);

ir::NodePtr create_as_strided_node(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset,
    bool is_out = false);

/* Debug API to dump memory stats of View Table. */
void dumpViewTableMemoryStat();

void flushWithMarkStep();
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
    at::TensorList tensors,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info = nullptr,
    std::vector<HbLazyTensor> out_hb_lazy_tensor = {});

template <class F, class... Ts, std::size_t... Is>
void for_each_in_tuple(
    std::tuple<Ts...>& tuple,
    F func,
    std::index_sequence<Is...>) {
  (void)(int[]){0, ((void)func(std::get<Is>(tuple)), 0)...};
}
template <class F, class... Ts>
void for_each_in_tuple(std::tuple<Ts...>& tuple, F func) {
  for_each_in_tuple(tuple, func, std::make_index_sequence<sizeof...(Ts)>());
}

template <class...>
struct conjunction : std::true_type {};

template <class B1>
struct conjunction<B1> : B1 {};

template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

template <typename Tuple>
struct is_tuple_of_tensor_ref;

template <typename... Ts>
struct is_tuple_of_tensor_ref<std::tuple<Ts...>>
    : conjunction<std::is_same<at::Tensor&, Ts>...> {};

// TODO: Ideally we want a variant of HABANA_ASSERT like
// TORCH_INTERNAL_ASSERT_DEBUG_ONLY

template <typename ReturnType, typename NodeConstruct = void>
class LazyOp {
 public:
  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::set<size_t> metadata_indices = {},
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{std::move(metadata_indices)},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{out_index},
        m_sbs_runner{SBSInterface::getSBSHandler(m_symbol.toQualString())} {
    set_inputs(inputs);
  }

  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{},
        m_sbs_runner{SBSInterface::getSBSHandler(m_symbol.toQualString())} {
    set_inputs(inputs);
  }

  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const std::function<
          std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
          out_shapes_fn,
      int out_index = 0) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{},
        m_out_index{out_index},
        m_sbs_runner{SBSInterface::getSBSHandler(m_symbol.toQualString())} {
    if (out_shapes_fn) {
      m_out_shapes = out_shapes_fn(inputs, false);
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
            m_node ? m_node->op().toQualString() : "")} {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        std::is_class<NodeConstruct>::value,
        "This constructor is valid only when NodeConstruct is a class.");
    set_inputs(inputs);
  }

  explicit LazyOp(
      ir::NodePtr node,
      const std::vector<at::IValue>& inputs,
      std::set<size_t> metadata_indices,
      std::vector<std::vector<int64_t>> out_shapes = {},
      int out_index = 0)
      : m_node{std::move(node)},
        m_metadata_indices{std::move(metadata_indices)},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{out_index},
        m_sbs_runner{SBSInterface::getSBSHandler(
            m_node ? m_node->op().toQualString() : "")} {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        std::is_class<NodeConstruct>::value,
        "This constructor is valid only when NodeConstruct is a class.");
    set_inputs(inputs);
  }

  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      const at::TensorList& output_meta_tensors) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_out_index{},
        m_out_meta_tensors{output_meta_tensors},
        m_sbs_runner{SBSInterface::getSBSHandler(m_symbol.toQualString())} {
    set_inputs(inputs);

    for (const auto& out : m_out_meta_tensors) {
      m_out_shapes.emplace_back(out.sizes().vec());
    }
  }

  explicit LazyOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      std::vector<std::vector<int64_t>> out_shapes,
      const c10::ScalarType scalar_type) noexcept
      : m_symbol{at::Symbol::fromQualString(qualstring)},
        m_metadata_indices{},
        m_out_shapes{std::move(out_shapes)},
        m_out_index{},
        m_scalar_type(scalar_type),
        m_sbs_runner{SBSInterface::getSBSHandler(m_symbol.toQualString())} {
    set_inputs(inputs);
  }

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
    tensors.reserve(std::tuple_size<T>::value);
    for_each_in_tuple(results, [&hl_results, &tensors](const auto& result) {
      auto hl_result = GetHbLazyTensor(result);
      tensors.push_back(result);
      hl_results.push_back(hl_result);
    });

    if (isOptimizedLazyEager == false) {
      PT_LAZY_DEBUG("Normal Lazy Eager Path Chosen");
      auto node = create_node();
      for (auto hl_result : hl_results) {
        ir::Value& out = hl_result.CurrentIrValue();
        out.SetNode(
            node,
            hl_result.GetDevice(),
            hl_result.GetSizes(),
            hl_result.dtype_optional(),
            i);
        updateDstDependencies(hl_result, tensors[i], false);
        i++;
      }
    } else {
      PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
      std::vector<ir::Value> input_vals = prepare_lazy_eager_input_values();
      info_to_lazy_backend->set_input_values(input_vals);
    }

    runSBS(tensors);
    flush_op(tensors, info_to_lazy_backend, hl_results);
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<not is_tuple_of_tensor_ref<T>::value, T>::type call() {
    PT_LAZY_DEBUG("Lazy Call not_Tuple_Of_Tensor :: ", m_symbol.toQualString());
    bool isView = false;
    isView = viewUpdateInputs();
    std::shared_ptr<HbLazyFrontEndInfoToBackend> infoToBackEnd =
        std::make_shared<HbLazyFrontEndInfoToBackend>();
    infoToBackEnd->set_lazy_op_name(m_symbol.toQualString());

    auto context = habana_lazy_executor.getDeviceExecutionContext(0);

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
  typename std::enable_if<is_tuple_of_tensor_ref<T>::value, T>::type HandleLazy(
      T results,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend =
          nullptr) {
    const auto& node = create_node();
    int i = 0;
    std::vector<at::Tensor> tensors;
    tensors.reserve(std::tuple_size<T>::value);
    const auto& out_shapes = m_out_shapes;
    auto context = habana_lazy_executor.getDeviceExecutionContext();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        out_shapes.size() == std::tuple_size<T>::value);

    for_each_in_tuple(
        results,
        [&node, &i, &tensors, out_shapes, context](const auto& result) {
          auto hl_result = GetHbLazyTensor(result);
          tensors.push_back(result);
          updateDstDependencies(hl_result, result, true);
          ir::Value& out = hl_result.CurrentIrValue();
          out.SetNode(
              node,
              hl_result.GetDevice(),
              hl_result.GetSizes(),
              hl_result.dtype_optional(),
              i);
          const auto& out_shape = out_shapes.at(i);
          if (result.sizes() != out_shape) {
            auto impl = hl_result.getAttachedTensorImpl();
            THHTensor_resizeNd(
                impl, out_shape.size(), out_shape.data(), nullptr);
            result.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
          }
          context->MarkTensorStatus(
              hl_result.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
          ++i;
        });
    runSBS(tensors);
    flush_op(tensors, info_to_lazy_backend);
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<is_tuple_of_tensor_ref<T>::value, T>::type call(
      T results) {
    PT_LAZY_DEBUG("Lazy Call Tuple_Of_Tensor :: ", m_symbol.toQualString());
    bool isView = false;
    isView = viewUpdateInputs();
    std::shared_ptr<HbLazyFrontEndInfoToBackend> infoToBackEnd =
        std::make_shared<HbLazyFrontEndInfoToBackend>();
    infoToBackEnd->set_lazy_op_name(m_symbol.toQualString());

    auto context = habana_lazy_executor.getDeviceExecutionContext(0);

    // Temporarily disabled the switch - To Do
    if (is_optimized_lazy_eager_supported(
            isView, context->viewContext.isLazyViewPresent) &&
        false) {
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
    viewUpdateInputs();

    const auto& node = create_node();
    const auto& t = get_inputs().at(m_out_index).toTensor();
    const auto& result =
        empty_hpu_lazy(1, t.options(), t.suggest_memory_format(), false);
    auto hl_result = GetHbLazyTensor(result);
    ir::Value& out = hl_result.CurrentIrValue();
    out.SetNode(
        node,
        hl_result.GetDevice(),
        hl_result.GetSizes(),
        hl_result.dtype_optional());
    updateDstDependencies(hl_result, result, false);
    runSBS(result);
    return result.item().template to<T>();
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_void<T>::value, T>::type call(
      at::TensorList tensors) {
    auto context = habana_lazy_executor.getDeviceExecutionContext();
    const auto& node = create_node();
    int i = 0;

    for (const auto& tensor : tensors) {
      auto hl_result = GetHbLazyTensor(tensor);
      updateDstDependencies(hl_result, tensor, true);
      hl_result.CurrentIrValue().SetNode(
          node,
          hl_result.GetDevice(),
          hl_result.GetSizes(),
          hl_result.dtype_optional(),
          i++);
      context->MarkTensorStatus(
          hl_result.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
    }
    runSBS(tensors);
    flush_op(tensors);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, std::vector<at::Tensor>>::value, T>::
      type
      call() {
    const auto& tensors = get_result_overrideable();
    const auto& node = create_node();
    int i = 0;

    for (const auto& tensor : tensors) {
      auto hl_result = GetHbLazyTensor(tensor);
      hl_result.CurrentIrValue().SetNode(
          node,
          hl_result.GetDevice(),
          hl_result.GetSizes(),
          hl_result.dtype_optional(),
          i++);
      updateDstDependencies(hl_result, tensor, false);
    }
    runSBS(tensors);
    flush_op(tensors);

    return tensors;
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
    auto hl_result = GetHbLazyTensor(result);
    if (isOptimizedLazyEager == false) {
      PT_LAZY_DEBUG("Normal Lazy Eager Path Chosen");
      const auto& node = create_node();
      ir::Value& out = hl_result.CurrentIrValue();
      out.SetNode(
          node,
          hl_result.GetDevice(),
          hl_result.GetSizes(),
          hl_result.dtype_optional());
      updateDstDependencies(hl_result, result, false);
    } else {
      PT_LAZY_DEBUG("Optimized Lazy Eager Path Chosen");
      std::vector<ir::Value> input_vals = prepare_lazy_eager_input_values();
      info_to_lazy_backend->set_input_values(input_vals);
    }

    runSBS(result);
    flush_op(result, info_to_lazy_backend, {hl_result});
    return result;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type call() {
    PT_LAZY_DEBUG("Lazy Call :: ", m_symbol.toQualString());
    bool isView = false;
    isView = viewUpdateInputs();
    std::shared_ptr<HbLazyFrontEndInfoToBackend> infoToBackEnd =
        std::make_shared<HbLazyFrontEndInfoToBackend>();
    infoToBackEnd->set_lazy_op_name(m_symbol.toQualString());

    auto context = habana_lazy_executor.getDeviceExecutionContext(0);

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

  bool viewUpdateInputsProcessSingleTensor(at::Tensor& t, size_t& idx) {
    bool is_view = false;
    auto context = habana_lazy_executor.getDeviceExecutionContext();
    if (t.defined() && (t.device().type() == c10::DeviceType::HPU)) {
      auto hl_t = GetHbLazyTensor(t);

      // if it is base tensor, use the most recent version else check if
      // it is a view
      auto id = hl_t.getTensorUniqueId();
      {
        std::lock_guard<std::recursive_mutex> view_table_lock(
            context->viewContext.GetViewTableMutex());
        c10::optional<at::Tensor> base_tensor =
            context->viewContext.GetOrigTensorMapEntry(id);
        if (base_tensor != c10::nullopt) {
          m_inputs[idx] = base_tensor;
        } else {
          if (HbLazyTensorViews::HandleViews(t, hl_t)) {
            is_view = true;
          }
        }
      }
    } // if (t.defined() && (

    return is_view;
  }

  bool viewUpdateInputs() {
    size_t idx = 0;
    bool is_view = false;
    for (auto ival : m_inputs) {
      if (ival.isTensor()) {
        auto t = ival.toTensor();
        is_view = viewUpdateInputsProcessSingleTensor(t, idx);
      } else if (ival.isTensorList()) {
        auto tl = ival.toTensorVector();
        for (size_t i = 0; i < tl.size(); ++i) {
          auto& t = tl[i];
          if (viewUpdateInputsProcessSingleTensor(t, idx)) {
            is_view = true;
          }
        } // for( size_t
      }

      idx++;
    }
    return is_view;
  }

  void HandleViewsInplace(
      const at::Tensor& self,
      habana_lazy::HbLazyTensor& hl_self) {
    auto out_t = empty_hpu_lazy(
        self.sizes(), self.options(), self.suggest_memory_format(), false);

    if (!is_inplace(m_symbol)) {
      // out variant needs storage as it is a graph input
      out_t = empty_hpu_lazy(
          self.sizes(), self.options(), self.suggest_memory_format(), true);
      for (size_t idx = m_inputs.size() - 1; idx >= 0; idx--) {
        auto t = m_inputs[idx];
        if (t.isTensor() && t.toTensor().is_same(self)) {
          m_inputs[idx] = out_t;
          // break after first update because we can cases like torch.ge(a, b,
          // out = a). In this case need to replace only out = a case
          break;
        }
      }
    }

    const auto& node = create_node();

    hl_self = GetHbLazyTensor(out_t);
    ir::Value& out = hl_self.CurrentIrValue();
    out.SetNode(
        node,
        hl_self.GetDevice(),
        hl_self.GetSizes(),
        hl_self.dtype_optional());

    flush_op(out_t);
    // add strided insert node and update most recent version of original
    // tensor
    strided_insert_hpu_lazy(self, out_t);
  }

  // For inplace/out variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type
  HandleLazy(
      at::Tensor& self,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend =
          nullptr) {
    auto context = habana_lazy_executor.getDeviceExecutionContext();
    auto hl_self = GetHbLazyTensor(self);

    auto id = hl_self.getTensorUniqueId();
    auto is_self_view = (context->viewContext.GetViewTableEntry(id) != nullptr);

    std::vector<at::IValue> sbs_stack;
    // special handling for self tensor
    if (is_self_view == false) {
      // use most recent version of the tensor if applicable
      auto self_updated = HbLazyTensorViews::get_recent_base_tensor(self);
      hl_self = GetHbLazyTensor(self_updated);

      // identify the inplace index and replace it with updated version
      // m_inputs will be used in create_node()
      for (size_t idx = 0; idx < m_inputs.size(); idx++) {
        auto t = m_inputs[idx];
        if (t.isTensor() && t.toTensor().is_same(self)) {
          m_inputs[idx] = self_updated;
        }
      }
      // special handling for self tensor
      // skip ctrl edges for inplace
      // TODO do the same for out variants
      if (!is_inplace(m_symbol)) {
        updateDstDependencies(hl_self, self_updated, true);
      }
      const auto& node = create_node();
      ir::Value& out = hl_self.CurrentIrValue();
      out.SetNode(
          node,
          hl_self.GetDevice(),
          hl_self.GetSizes(),
          hl_self.dtype_optional());

      // Special handling for SBS in inplace, before the inplace op will
      // override the tensor
      if (is_inplace(m_symbol)) {
        m_sbs_runner->populateInputForCPUOp(
            get_inputs(), node->GetMetaData(), sbs_stack);
      }
    } else {
      HandleViewsInplace(self, hl_self);
    }

    // numel == 0 is the correct check, need the size check until pytorch
    // fixes it properly
    // https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
    auto out_shape = m_out_shapes.empty()
        ? get_inputs().at(m_out_index).toTensor().sizes().vec()
        : m_out_shapes[0];
    if (self.sizes() != out_shape) {
      auto impl = hl_self.getAttachedTensorImpl();
      THHTensor_resizeNd(impl, out_shape.size(), out_shape.data(), nullptr);
      self.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
    }

    context->MarkTensorStatus(
        hl_self.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);

    runSBS(self, sbs_stack);
    flush_op(self, info_to_lazy_backend);
    return self;
  }

  // For inplace/out variants
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type call(
      at::Tensor& self) {
    PT_LAZY_DEBUG("Lazy Call Inplace:self :: ", m_symbol.toQualString());
    bool isView = false;

    // Handle views or fetch updated tensor for all the inputs
    isView = viewUpdateInputs();

    std::shared_ptr<HbLazyFrontEndInfoToBackend> infoToBackEnd =
        std::make_shared<HbLazyFrontEndInfoToBackend>();
    infoToBackEnd->set_lazy_op_name(m_symbol.toQualString());

    auto context = habana_lazy_executor.getDeviceExecutionContext(0);

    // Temporarily disabled the switch - To Do
    if (is_optimized_lazy_eager_supported(
            isView, context->viewContext.isLazyViewPresent) &&
        false) {
      size_t lazy_eager_key = 0;
      bool IsOptimizedLazyEagerCached =
          calculate_key_and_check_optimized_lazy_eager_cache(lazy_eager_key);

      infoToBackEnd->set_optimized_lazy_eager_key(lazy_eager_key);
      infoToBackEnd->set_is_optimized_lazy_eager(IsOptimizedLazyEagerCached);
    }

    return HandleLazy(self, infoToBackEnd);
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, T>::type
  HandleLazy(
      std::shared_ptr<HbLazyFrontEndInfoToBackend> info_to_lazy_backend =
          nullptr) {
    const at::Tensor& self = get_inputs().at(0).toTensor();
    auto hl_self = GetHbLazyTensor(self);
    const auto& node = create_node();
    ir::Value& out = hl_self.CurrentIrValue();
    out.SetNode(
        node,
        hl_self.GetDevice(),
        hl_self.GetSizes(),
        hl_self.dtype_optional());
    updateDstDependencies(hl_self, self, true);

    auto out_shape = m_out_shapes.empty()
        ? get_inputs().at(m_out_index).toTensor().sizes().vec()
        : m_out_shapes[0];
    if (self.sizes() != out_shape) {
      auto impl = hl_self.getAttachedTensorImpl();
      THHTensor_resizeNd(impl, out_shape.size(), out_shape.data(), nullptr);
      self.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape);
    }

    auto context = habana_lazy_executor.getDeviceExecutionContext();
    context->MarkTensorStatus(
        hl_self.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
    runSBS(self);
    flush_op(self, info_to_lazy_backend);
    return self;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, const at::Tensor&>::value, T>::type
  call() {
    PT_LAZY_DEBUG("Lazy Call Inplace :: ", m_symbol.toQualString());
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

    return HandleLazy(infoToBackEnd);
  }

 private:
  bool isMetadataCandidate(const at::IValue& input) const {
    return input.isBool() || input.isDevice() || input.isIntList() ||
        input.isDoubleList() || input.isBoolList() || input.isString() ||
        input.isNone() ||
        (input.isList() &&
         !input.toList().elementType()->cast<at::TensorType>());
  }

  template <typename T = ReturnType>
  typename std::enable_if<not is_tuple_of_tensor_ref<T>::value, T>::type
  get_result() {
    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        std::tuple_size<T>::value == m_out_shapes.size());

    unsigned i = 0;
    ReturnType results;

    for_each_in_tuple(results, [&](auto& result) {
      auto t = get_inputs().at(m_out_index).toTensor();
      result = empty_hpu_lazy(
          m_out_shapes[i++], t.options(), t.suggest_memory_format(), false);
    });
    return results;
  }

  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor>::value, T>::type
  get_result() {
    // Get results from derived class when index is negative
    if (m_out_index < 0) {
      return get_result_overrideable();
    }

    if (m_out_meta_tensors.empty()) {
      const auto& t = get_inputs().at(m_out_index).toTensor();
      const auto& out_shape =
          m_out_shapes.empty() ? t.sizes() : m_out_shapes[0];
      if (m_scalar_type != c10::ScalarType::Undefined) {
        return empty_hpu_lazy(
            out_shape,
            t.options().dtype(m_scalar_type),
            t.suggest_memory_format(),
            false);
      } else {
        return empty_hpu_lazy(
            out_shape, t.options(), t.suggest_memory_format(), false);
      }
    }

    const auto& t = m_out_meta_tensors[0];
    if (m_scalar_type != c10::ScalarType::Undefined) {
      return empty_hpu_lazy(
          t.sizes(),
          t.options().dtype(m_scalar_type),
          t.suggest_memory_format(),
          false);
    } else {
      return empty_hpu_lazy(
          t.sizes(), t.options(), t.suggest_memory_format(), false);
    }
  }

  void create_inputs(
      ir::ValueList& values,
      std::vector<at::Tensor>& input_pt_vec,
      ir::MetaData& metadata,
      bool is_optimized_lazy_eager) {
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      const at::IValue& input = m_inputs[i];
      if (m_metadata_indices.count(i)) {
        // Already taken care in optimized lazy eager JIT graph key calculation
        // so not required for the optimized lazy eager.
        if (!is_optimized_lazy_eager) {
          metadata.set(input, i);
        }
        continue;
      }

      if (input.isScalar()) {
        // Already taken care in optimized lazy eager JIT graph key calculation
        // so not required for the optimized lazy eager
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
          auto val = GetHbLazyTensor(t).GetIrValue();
          if (!is_optimized_lazy_eager) {
            values.emplace_back(val);
            input_pt_vec.emplace_back(t);
          } else {
            // Taking care of duplicate values here itself for optimized lazy
            // eager. In normal flow it is taken care later in the flow. To Do -
            // To make it same for normal flow as well.
            auto it = find(values.begin(), values.end(), val);
            if (it == values.end()) {
              values.emplace_back(val);
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
              opt_tensors.emplace_back(GetHbLazyTensor(t).GetIrValue());
              list_input_pt_vec.emplace_back(t);
            } else {
              // Taking care of duplicate values here itself for optimized lazy
              // eager. In normal flow it is taken care later in the flow. To Do
              // - To make it same for normal flow as well.
              auto val = GetHbLazyTensor(t).GetIrValue();
              auto it = find(values.begin(), values.end(), val);
              if (it == values.end()) {
                values.emplace_back(val);
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

  void set_inputs(const std::vector<at::IValue>& inputs) {
    auto inputsHpu = inputs;
    for (auto& t : inputsHpu) { // Any tensor on CPU needs to be moved to HPU
      if (t.isTensor() && t.toTensor().defined() &&
          t.toTensor().device().type() != c10::DeviceType::HPU) {
        const at::Tensor& tensor = t.toTensor();
        at::Tensor tinput;
        // If the CPU tensor is a wrapped number, then use
        // get_tensor_for_scalar method to retrieve cached HPU tensors for
        // the scalar value
        if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
          // is_wrapped_number: True if a tensor was auto-wrapped from a
          // C++ or Python number.
          auto dtype = tensor.scalar_type();
          tinput = get_tensor_for_scalar(
              tensor.item().toDouble(), at::TensorOptions().dtype(dtype));
        } else {
          // Use non_blocking .to()
          tinput = tensor.to(c10::kHPU, true);
        }
        t = c10::IValue(tinput);
      }
    }
    m_sbs_runner->setCPUInputs(inputsHpu);
    m_inputs = inputsHpu;
  }

  const std::vector<std::vector<int64_t>>& get_out_shapes() const {
    return m_out_shapes;
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

  void set_scalar_type(const c10::ScalarType scalar_type) {
    m_scalar_type = scalar_type;
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
    if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
      auto& device = synapse_helpers::HPURegistrar::get_device();
      optimized_key =
          at::hash_combine(optimized_key, device.getDeterministic());
    }
    std::unordered_set<size_t> input_hash_values;
    for (size_t i = 0; i < m_inputs.size(); ++i) {
      optimized_key = at::hash_combine(optimized_key, i);
      const at::IValue& input = m_inputs[i];
      // Create stack based on input tensors / tensor lists.
      // Metadata and scalars are part of key calculation, so we skip them.
      if (m_metadata_indices.count(i)) {
        if (input.isList()) {
          for (auto& v : input.toListRef()) {
            optimized_key =
                at::hash_combine(optimized_key, at::IValue::hash(v));
          }
        } else {
          optimized_key =
              at::hash_combine(optimized_key, at::IValue::hash(input));
        }
        continue;
      } else if (input.isScalar()) {
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
        // Not handled so returning null key
        optimized_key = 0;
        break;
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
        IsCached = habana_lazy::OptimizedLazyGraphCache::GetOptimizedLazyCache()
                       .IsCached(lazy_eager_key);
      }
    }

    return IsCached;
  }

  std::vector<ir::Value> prepare_lazy_eager_input_values() {
    ir::ValueList values;
    std::vector<at::Tensor> input_pt_vec;
    ir::MetaData metadata;

    create_inputs(values, input_pt_vec, metadata, true);
    return values;
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
    ir::ValueList values;
    std::vector<at::Tensor> input_pt_vec;
    ir::MetaData metadata;

    create_inputs(values, input_pt_vec, metadata, false);
    auto node = ir::Node::Create(m_symbol, values);

    if (metadata.size()) {
      node->SetMetaData(metadata);
    }

    node->AddInputPtTensors(input_pt_vec);

    return node;
  }

 private:
  std::vector<at::Tensor> m_input_pt_tensors;
  ir::NodePtr m_node = nullptr;
  const at::Symbol m_symbol;
  const std::set<size_t> m_metadata_indices;
  std::vector<std::vector<int64_t>> m_out_shapes;
  const int m_out_index;
  at::TensorList m_out_meta_tensors = {};
  std::vector<at::IValue> m_inputs = {};
  c10::ScalarType m_scalar_type = c10::ScalarType::Undefined;
  const std::shared_ptr<SBSInterface> m_sbs_runner;
  void update_hash_key_for_tensor(const at::Tensor& t, size_t& optimized_key) {
    auto hl_tensor = TryGetHbLazyTensor(t);
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
      if (val.mp_node && !(val.mp_node->is_input())) {
        optimized_key = 0;
      }
    }
  }
};

template <typename T>
class LazyOpWithTypePromotion : public LazyOp<T> {
 public:
  explicit LazyOpWithTypePromotion(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      bool is_outfn,
      bool safe_cast_check_,
      const std::function<
          std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
          out_shapes_fn = nullptr);

 private:
  habana_helpers::DTypeHelper dtype_helper_;

  T get_result_overrideable() override;
};

template <typename T>
class PromoteIntToFloat : public LazyOp<T> {
 public:
  explicit PromoteIntToFloat(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      bool is_outfn,
      bool safe_cast_check_,
      const std::function<
          std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
          out_shapes_fn = nullptr);

 private:
  habana_helpers::DTypeHelper dtype_helper_;

  T get_result_overrideable() override;
};

template <typename ReturnType>
class LazyBinaryOp : public LazyOp<ReturnType> {
 public:
  explicit LazyBinaryOp(
      const std::string& qualstring,
      const std::vector<at::IValue>& inputs,
      bool is_outfn,
      bool safe_cast_check,
      const std::set<size_t>& metadata_indices = {},
      const std::vector<std::vector<int64_t>>& out_shapes = {},
      int out_index = 0)
      : LazyOp<ReturnType>(
            qualstring,
            inputs,
            metadata_indices,
            out_shapes,
            out_index),
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
      inputs[i] = self;
    }

    if (inputs_updated) {
      LazyOp<T>::set_inputs(inputs);
    }

    auto results = LazyOp<T>::call();
    return results;
  }

  // For inplace binary, the promoted type takes the self's type
  template <typename T = ReturnType>
  typename std::enable_if<std::is_same<T, at::Tensor&>::value, T>::type call(
      at::Tensor& self) {
    auto inputs = LazyOp<T>::get_inputs();

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
      inputs[i] = self;
    }

    if (inputs_updated) {
      LazyOp<T>::set_inputs(inputs);
    }

    return LazyOp<T>::call(self);
  }

 private:
  c10::ScalarType dst_dtype_ = c10::ScalarType::Undefined;
  bool is_outfn_ = false;
  bool safe_cast_check_ = false;

  ReturnType get_result_overrideable() override;
};

} // namespace habana_lazy
