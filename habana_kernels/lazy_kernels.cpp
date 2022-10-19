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
#include <ATen/InferSize.h>
#include <c10/core/SymIntArrayRef.h>
#include <cstdlib>
#include <ctime>
#include <utility>
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_composite_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/conv_kernels.h"
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/embedding_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_kernels/loss_kernels.h"
#include "habana_kernels/lowering_util.h"
#include "habana_kernels/nonzero_kernel.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/pool_kernels.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/reduction2_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/repeat.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/softmax_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/triangular_kernels.h"
#include "habana_kernels/upsample_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/ops/cast_ops.h"
#include "habana_lazy/ops/cat.h"
#include "habana_lazy/ops/constant.h"
#include "habana_lazy/ops/convolution.h"
#include "habana_lazy/ops/custom_op.h"
#include "habana_lazy/ops/embedding.h"
#include "habana_lazy/ops/embedding_bag.h"
#include "habana_lazy/ops/hpu_input.h"
#include "habana_lazy/ops/index.h"
#include "habana_lazy/ops/loss.h"
#include "habana_lazy/ops/matmul.h"
#include "habana_lazy/ops/mse_loss.h"
#include "habana_lazy/ops/norm.h"
#include "habana_lazy/ops/optimizer.h"
#include "habana_lazy/ops/pool.h"
#include "habana_lazy/ops/reduce_ops.h"
#include "habana_lazy/ops/shape_ops.h"
#include "habana_lazy/ops/tensor_shape.h"
#include "habana_lazy/ops/unpack.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/sbs_debug.h"
#include "habana_lazy/view.h"
#include "habana_lazy/view_utils.h"
#include "hpu_ops/cpu_fallback.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/habana_helpers/dtype_helpers.h"
#include "pytorch_helpers/pt_ver/torch_params_shim.h"
#include "pytorch_helpers/synapse_helpers/util.h"

using namespace habana;
using namespace at;

namespace habana_lazy {
static std::vector<int64_t> device_shape_tensor_size = {SYN_MAX_TENSOR_DIM};

static bool is_nonempty_tensor(const at::Tensor& tensor) {
  return tensor.dim() != 1 || tensor.size(0) != 0;
}

bool is_inplace(at::Symbol symbol) {
  bool is_inplace = false;

  auto node_name = symbol.toQualString();
  /*
  Since as_strided_lazy is now out of place op, we need a control edge to create
  new tensor for fill to avoid GC error
  %5 : Float(*, requires_grad=0,
  device=hpu:0) = hpu::as_strided_lazy(%id:3, %2, %3, %4)
  %6 : Float(*, requires_grad=0, device=hpu:0) = aten::fill_(%5, %1)
  */

  // TODO think of better way to avoid these string comparisons for multiple ops
  bool is_normal_inplace =
      (strcmp(node_name, "aten::fill_") && strcmp(node_name, "hpu::uniform_") &&
       strcmp(node_name, "hpu::random_") && strcmp(node_name, "hpu::normal_") &&
       strcmp(node_name, "hpu::geometric_") &&
       strcmp(node_name, "hpu::bernoulli_"));

  if (is_normal_inplace) {
    size_t len = strlen(node_name);
    char endch = node_name[len - 1];

    if (endch == '_') {
      is_inplace = true;
    }
  }
  return is_inplace;
}

bool to_lower_as_strided() {
  return GET_ENV_FLAG_NEW(PT_HPU_LOWER_AS_STRIDED);
}

void dumpViewTableMemoryStat() {
  PT_LAZY_TRACE;
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);

  PT_VIEWTABLE_DEBUG(
      "[ViewTable MemStats] #view_table map size: ",
      context->viewContext.viewTableSize(),
      ", total bytes: ",
      context->viewContext.viewTableBytes());

  PT_VIEWTABLE_DEBUG(
      "[ViewTable MemStats] #orig_tensor_map map size: ",
      context->viewContext.tensorMapSize(),
      ", total bytes: ",
      context->viewContext.tensorMapBytes());
}

void flushWithMarkStep() {
  // Generate a random number and invoke the mark_step
  static std::once_flag flag;
  std::call_once(flag, [&]() { srand((unsigned)time(0)); });

  // Generate a random number between 1 - 100
  auto rand_num = rand() % 100 + 1;

  // By default, we want to trigger 50% of the time
  auto aggressiveness = 50;
  if (const auto envp =
          std::getenv("INTERNAL_PT_HPU_LAZY_MARK_STEP_TEST_TRIGGER")) {
    aggressiveness = std::stoul(envp, nullptr, 10);
    // Cap the trigger to at least 1% to at most 100%
    if (aggressiveness < 1) {
      aggressiveness = 0;
    } else if (aggressiveness > 100) {
      aggressiveness = 100;
    }
  }
  if (rand_num < aggressiveness) {
    PT_LAZY_DEBUG("Triggering a mark_step");
    HbLazyTensor::StepMarker({});
  }
}

// For the ops that don't use LazyOp to construct nodes.
// Remove when all ops move to LazyOp style.
void flush_op(
    UNUSED at::TensorList tensors,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info,
    std::vector<HbLazyTensor> out_hb_lazy_tensor) {
  const bool m_flush_op = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2;
  const bool m_random_flush = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 3;
  StageSubmission::getInstance().incrementAccumulatedOps();
  SBSDebug::getInstance().IncreaseOpsAndTensors(tensors.size());

  if (m_flush_op) {
    bool async =
        (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD) &&
         GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_EAGER_EXECUTION_THREAD));
    HbLazyTensor::StepMarker(
        {}, lazy_front_end_info, out_hb_lazy_tensor, async);
  } else if (m_random_flush) {
    flushWithMarkStep();
  } else if (StageSubmission::getInstance().isExceededMaxAccumlatedSize()) {
    PT_LAZY_DEBUG("Reached max accumulated graph size, triggering a mark_step");
    HbLazyTensor::StepMarker({}, lazy_front_end_info);
  }
}

template <typename SRC_DTYPE, typename DST_DTYPE>
inline void validateDownCast(const at::Tensor& src, ScalarType dstScalarType) {
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VALID_DATA_RANGE_CHECK)) {
    if (IsDefined(src) && src.numel() > 0) {
      auto max_int_val = (SRC_DTYPE)std::numeric_limits<DST_DTYPE>::max();
      auto min_int_val = (SRC_DTYPE)std::numeric_limits<DST_DTYPE>::lowest();
      auto src_detached = src.detach();
      auto src_max_val = src_detached.max().item().to<SRC_DTYPE>();
      auto src_min_val = src_detached.min().item().to<SRC_DTYPE>();
      bool condition = src_max_val <= max_int_val && src_min_val >= min_int_val;
      if constexpr (std::is_floating_point_v<SRC_DTYPE>) {
        if (!condition) {
          // When condition is not met lets try again without Nans and Infs
          // We can't eliminate them before first check as it causes performance
          // drop.
          //
          // Different approach was tried here
          // Performance results for double tensor with 1.000.000 elements
          // Averaged over 10 consecutive measurements
          //
          // CASE 1: No Nans and Infs special handling:
          // t = 810 us
          //
          // CASE 2: nan_to_num(c10::nullopt, max_int_val, min_int_val);
          // t = 2190 us (x2.7 with respect to CASE 1)
          //
          // CASE 3: In place nan_to_num_(c10::nullopt, max_int_val,
          // min_int_val);
          // t = 1050 us (x1.3 with respect to CASE 1)
          // It can't be used as it changes src tensor contents
          //
          // CASE 4: torch::where(torch::isfinite(src_detached), src_detached,
          // 0);
          // t = 4480 us (x5.5 with respect to CASE 1)
          //
          // CASE 5: manual min/max calculation in for loop
          // with nan/inf replacement, using data_ptr and numel
          // t = 2170 us for src.data_ptr()
          // t = 3200 us for src.detach().data_ptr()
          //
          // INF's have to be replaced by destination type extreme values not
          // source type. Source type extreme values can be out of range for
          // destination type and cause unwanted error.
          src_detached =
              src_detached.nan_to_num(c10::nullopt, max_int_val, min_int_val);
          src_max_val = src_detached.max().item().to<SRC_DTYPE>();
          src_min_val = src_detached.min().item().to<SRC_DTYPE>();
          condition = src_max_val <= max_int_val && src_min_val >= min_int_val;
        }
      }
      TORCH_CHECK(
          condition,
          "Error when trying to cast ",
          src.scalar_type(),
          " to ",
          dstScalarType,
          ", Input values range [",
          src_min_val,
          ", ",
          src_max_val,
          "] exceeds ",
          dstScalarType,
          " range [",
          min_int_val,
          ", ",
          max_int_val,
          "]");
    }
  } else {
    PT_LAZY_DEBUG(
        "Skipping validateDownCast from ",
        src.scalar_type(),
        " to ",
        dstScalarType);
  }
}

at::Tensor preProcessIfLongorDouble(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool& processed) {
  at::Tensor processed_tensor_cpu = src;
  c10::ScalarType old_type = src.scalar_type();
  c10::ScalarType new_type = src.scalar_type();
  // We need to cast data on CPU before copying if there is some unsupported
  // type
  if (src.scalar_type() == c10::ScalarType::Long) {
    validateDownCast<long, int>(src, c10::ScalarType::Int);
    processed_tensor_cpu = src.to(c10::ScalarType::Int);
    processed = true;
    old_type = c10::ScalarType::Long;
    new_type = c10::ScalarType::Int;
  } else if (src.scalar_type() == c10::ScalarType::Double) {
    validateDownCast<double, float>(src, c10::ScalarType::Float);
    processed_tensor_cpu = src.to(c10::ScalarType::Float);
    processed = true;
    old_type = c10::ScalarType::Double;
    new_type = c10::ScalarType::Float;
  }
  if (processed) {
    auto hl_tensor = GetOrCreateHbLazyTensor(dst, dst.device());
    if (dst.scalar_type() == c10::ScalarType::Long ||
        dst.scalar_type() == c10::ScalarType::Double) {
      hl_tensor.setTensorOriginalType(old_type);
      hl_tensor.SetScalarType(c10::make_optional(new_type));
    } else {
      hl_tensor.setTensorOriginalType(dst.scalar_type());
      hl_tensor.SetScalarType(c10::make_optional(dst.scalar_type()));
    }
  }
  return processed_tensor_cpu;
}

std::vector<int64_t> CalculateStrides5d(
    const IntArrayRef sizes,
    c10::MemoryFormat format) {
  HABANA_ASSERT(sizes.size() == 5);
  if (c10::MemoryFormat::ChannelsLast3d == format) {
    return {
        sizes[1] * sizes[2] * sizes[3] * sizes[4],
        1,
        sizes[1] * sizes[3] * sizes[4],
        sizes[1] * sizes[4],
        sizes[1]};
  }

  return {
      sizes[1] * sizes[2] * sizes[3] * sizes[4],
      sizes[4] * sizes[3] * sizes[2],
      sizes[4] * sizes[3],
      sizes[4],
      1};
}

std::vector<int64_t> CalculateStrides(
    const IntArrayRef sizes,
    c10::MemoryFormat format) {
  HABANA_ASSERT(sizes.size() == 4);
  if (c10::MemoryFormat::ChannelsLast == format) {
    return {sizes[1] * sizes[2] * sizes[3], 1, sizes[1] * sizes[3], sizes[1]};
  }

  return {sizes[1] * sizes[2] * sizes[3], sizes[3] * sizes[2], sizes[3], 1};
}

ir::Value AddControlEdge(const at::Tensor& src, const at::Tensor& dst) {
  auto hb_result = GetOrCreateHbLazyTensor(dst, dst.device());
  auto hb_tensor = GetOrCreateHbLazyTensor(src, src.device());
  // We are using this lazy tensor as output on some op
  // Version counter tracks the number of times we do that
  // if its zero, that means this tensor hasnt been output in any op
  hb_result.updateVersion();
  auto node = ir::Node::Create(
      Symbol::fromQualString("hpu::control_edge_other_"),
      {hb_tensor.GetIrValue(), hb_result.GetIrValue()});
  node->set_as_control_edge();
  std::vector<at::Tensor> input_pt_vec;
  input_pt_vec.push_back(src);
  input_pt_vec.push_back(dst);
  ir::Value& out = hb_result.CurrentIrValue();
  out.SetNode(
      node,
      hb_result.GetDevice(),
      hb_result.GetSizes(),
      hb_result.dtype_optional());
  node->AddInputPtTensors(input_pt_vec);
  return out;
}

void updateDstDependencies(
    HbLazyTensor& hl_dst,
    const Tensor& dst,
    bool in_place) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    return;
  };

  if (in_place) {
    // We are using this lazy tensor as output on some op
    // Version counter tracks the number of times we do that
    // if its zero, that means this tensor hasnt been output in any op
    hl_dst.updateVersion();
    auto hb_result = GetOrCreateHbLazyTensor(dst, dst.device());
    ir::Value val{hb_result.GetIrValue().m_data_ptr.lock()};
    auto node = ir::Node::Create(
        Symbol::fromQualString("hpu::control_edge_"), {hb_result.GetIrValue()});
    node->set_as_control_edge();
    std::vector<at::Tensor> input_pt_vec;
    input_pt_vec.push_back(dst);
    ir::Value& out = val;
    out.SetNode(
        node,
        hb_result.GetDevice(),
        hb_result.GetSizes(),
        hb_result.dtype_optional());
    hb_result.AssignIrValue(val);
    node->AddInputPtTensors(input_pt_vec);
  }
}

Tensor _copy_from_and_resize_lazy(const Tensor& self, const Tensor& dst) {
  auto sizes = self.sizes().vec();
  if (self.sizes() != dst.sizes()) {
    dst.resize_(self.sizes());
  }

  return dst.copy_(self);
}

// Here self corresponds to the output of op(view_tensor) where view_tensor =
// strided_view(base)
void strided_insert_hpu_lazy(
    const Tensor& self,
    const Tensor& insert_t,
    bool is_flush) {
  PT_LAZY_TRACE;
  auto hl_self = GetHbLazyTensor(self);
  auto id = hl_self.getTensorUniqueId();

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  LOCK_VIEW_TABLE_MUTEX(context->viewContext);
  StrideParams* params_ptr = context->viewContext.GetViewTableEntry(id);
  TORCH_CHECK(params_ptr != nullptr, "incorrect tensor id");

  if (params_ptr->optype == kStridedOpDefault) {
    context->viewContext.SetViewStatus(id, kViewWrite);
  }

  // pick the most recent version
  Tensor recent_orig_t =
      HbLazyTensorViews::get_recent_base_tensor(params_ptr->base);
  auto recent_insert_t = HbLazyTensorViews::get_recent_base_tensor(insert_t);
  // Incase of slice operator on multi axes, it comes as different
  // slice operation on differnt axes, we combine them into single slice
  // operation.
  std::vector<StridedOpSliceParams> back_to_back_slices;
  auto params_ptr_link = params_ptr;
  while (params_ptr_link && params_ptr_link->optype == kStridedOpSlice) {
    back_to_back_slices.push_back(params_ptr_link->params.slice_param);
    auto parent_id =
        GetHbLazyTensor(params_ptr_link->parent).getTensorUniqueId();
    params_ptr_link = context->viewContext.GetViewTableEntry(parent_id);
  }
  bool use_strided_insert = (params_ptr_link != nullptr);

  at::Tensor out;
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SLICE_INSERT) ||
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) ||
      use_strided_insert) {
    out = add_strided_insert_node(
        recent_orig_t,
        recent_insert_t,
        params_ptr->strides,
        params_ptr->offset,
        is_flush);
  } else {
    out = add_slice_insert_node(
        recent_orig_t, recent_insert_t, back_to_back_slices);
  }
  // update orig tensor map
  auto param_id = GetHbLazyTensor(params_ptr->base).getTensorUniqueId();
  context->viewContext.AddOrigTensorMapEntry(param_id, out);

  PT_VIEWTABLE_DEBUG("orig tensor map entry created for ", param_id);
  return;
}

/* checks if fallback to original op is possible*/
bool is_fallback_original_op(const Tensor& self, const Tensor& out) {
  PT_LAZY_TRACE;
  // PT_HPU_FCD_STRIDE_OPT is disabled by default as a workaround for
  // transformer accuracy issues. Disabling this flag will merge consecutive
  // strided ops to single strided_view op. Enable this flag for improving the
  // perf in case of back to back as_strided ops.
  if (GET_ENV_FLAG_NEW(PT_HPU_FCD_STRIDE_OPT)) {
    bool is_fallback = true;
    auto context = habana_lazy_executor.getDeviceExecutionContext(0);

    // trace until the base tensor is reached and check if there are any
    // as_strided ops fall back not possible if there are as_strided ops in the
    // sequence.
    auto self_id = GetHbLazyTensor(self).getTensorUniqueId();
    {
      LOCK_VIEW_TABLE_MUTEX(context->viewContext);
      StrideParams* params_ptr =
          context->viewContext.GetViewTableEntry(self_id);
      while (params_ptr != nullptr) {
        if (params_ptr->optype == kStridedOpDefault) {
          is_fallback = false;
          break;
        }

        auto parent_id =
            GetHbLazyTensor(params_ptr->parent).getTensorUniqueId();
        params_ptr = context->viewContext.GetViewTableEntry(parent_id);
      }
    }
    return is_fallback;
  } else {
    auto hb_result = GetHbLazyTensor(out);
    auto context = habana_lazy_executor.getDeviceExecutionContext(0);
    LOCK_VIEW_TABLE_MUTEX(context->viewContext);
    auto strided_param = HbLazyTensorViews::getViewTableParams(hb_result);
    return (
        GetHbLazyTensor(self).getTensorUniqueId() ==
        GetHbLazyTensor(strided_param->base).getTensorUniqueId());
  }
}

at::Tensor append_to_batch_h2d_list(const at::Tensor& scalar_tensor) {
  const auto& context = habana_lazy_executor.getDeviceExecutionContext(0);

  const auto& t =
      empty_hpu_lazy({}, scalar_tensor.options(), c10::nullopt, true);
  t.unsafeGetTensorImpl()->set_wrapped_number(true);

  bool processed = false;
  const auto& tensor = preProcessIfLongorDouble(scalar_tensor, t, processed);

  // Mark as input
  HbLazyTensor hb_tensor = GetHbLazyTensor(t);
  setTensorAsInputNode(hb_tensor);
  context->MarkTensorStatus(
      hb_tensor.getDataPtr(), LazyTensorExecutionStatus::kINPUT);

  auto internal_tensor = hb_tensor.GetHbLazyTensorData().value();
  internal_tensor.unsafeGetTensorImpl()->set_wrapped_number(true);

  // Actual Copy is done during JIT graph creation/lowering
  context->copy_scalar_to_hpu_tensor_list.emplace_back(tensor, internal_tensor);

  return t;
}

/**
 * Returns a tensor for a scalar value.
 * In case of 64b dtypes such as Long/Double it returns a tensor
 * where the FE (user's PT tensor) is in Long/Double and the BE (device storage)
 * is in Int/Float
 */
at::Tensor get_tensor_for_scalar(
    double alpha,
    const at::TensorOptions& options) {
  at::Tensor alpha_tensor;

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  static uint64_t hit_count, miss_count;

  auto map_it = context->scalar_to_tensor_map.find(
      std::make_pair(alpha, options.dtype().toScalarType()));
  if (map_it == context->scalar_to_tensor_map.end()) {
    if (false == GET_ENV_FLAG_NEW(PT_HPU_SCALAR_H2D_COPY_MULTIPLE)) {
      alpha_tensor = at::tensor(alpha).to(options.dtype()).to(c10::kHPU, true);
    } else {
      alpha_tensor =
          append_to_batch_h2d_list(at::tensor(alpha).to(options.dtype()));
    }

    // Add to scalar value to device tensor cache
    context->scalar_to_tensor_map[std::make_pair(
        alpha, options.dtype().toScalarType())] = alpha_tensor;
    PT_LAZY_DEBUG(
        "scalar_to_tensor_map #miss: ",
        ++miss_count,
        " alpha = ",
        alpha,
        " map size = ",
        context->scalar_to_tensor_map.size());
  } else {
    alpha_tensor = map_it->second;
    PT_LAZY_DEBUG(
        "scalar_to_tensor_map #hit: ",
        ++hit_count,
        " alpha = ",
        alpha,
        " map size = ",
        context->scalar_to_tensor_map.size());
  }

  return alpha_tensor;
}

Tensor& copy_hpu_lazy_D2D(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;
  if (!habana_helpers::is_supported_type(self.scalar_type())) {
    // only dst can be unsupported dtype since copy_h2d and empty_hpu calls
    // would fallback to cpu for unsupported dtypes
    // also note that since the dst is an unsupported dtype, we will move
    // the dst back to cpu with this copy
    PT_LAZY_WARN(
        "Falling back to CPU - Unsupported dst type in D2D copy: src : ",
        src.scalar_type(),
        ", dst: ",
        self.scalar_type());
    auto fb_self = self.cpu();
    auto fb_src = src.cpu();
    return fb_self.copy_(fb_src);
  }

  ir::NodePtr node;
  std::vector<at::Tensor> input_pt_vec;
  // pick the most recent version of src tensor
  Tensor src_updated = HbLazyTensorViews::get_recent_base_tensor(src);
  HbLazyTensor hb_tensor =
      GetOrCreateHbLazyTensor(src_updated, src_updated.device());
  auto hlresult = GetOrCreateHbLazyTensor(self, src_updated.device());
  auto layout_format = hb_tensor.GetTensorLayout();
  hlresult.SetTensorLayout(layout_format);
  /* We can't create a long/double target in the device. Even a cast will not
    work as these data types are not available within the device. The only way
    to make progress is to just do a normal D2D so that the target will also
    be the same as source, and when we want to pull this out to CPU, the D2H
    will handle the type conversion*/
  if ((self.scalar_type() == c10::ScalarType::Long) ||
      (self.scalar_type() == c10::ScalarType::Double) ||
      (src_updated.dtype() == self.dtype())) {
    if (hb_tensor.IsExecutionInProgress()) {
      auto context =
          habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
      context->JoinPendingLaunchThread();
    }
    // If both src and dst are already processed ,  go and do the DMA dont
    // wait Else , If we already have storage in dst, add memcopy node to lazy
    // graph and we want to copy to existing tensor and not a new one
    // Kernel expects us to pass dst as second input in that case
    auto src_id = hb_tensor.getTensorUniqueId();
    auto dst_id = hlresult.getTensorUniqueId();
    if (src_id == dst_id) {
      return self;
    }

    // graph cycle happens in squad 8x with view table mechanism
    // %id:3646 = hpu::as_strided_lazy(%id:18.1, %89, %90, %91)
    // %id:18 = hpu::habana_d2d_memcpy_other(%id:3646, %id:18.1)
    auto src_parent = HbLazyTensorViews::get_base_tensor(src_updated);
    auto src_parent_id = GetHbLazyTensor(src_parent).getTensorUniqueId();

    if (src_parent_id == dst_id) {
      return self;
    }

    // Handle views and lhs slice
    auto is_view = HbLazyTensorViews::HandleViewsD2D(src, self);
    if (is_view == false) {
      AddMemcpy(src_updated, self);
    }
  } else {
    node = std::make_shared<ir::Cast>(src, self.scalar_type(), non_blocking);

    auto context = habana_lazy_executor.getDeviceExecutionContext(0);
    auto id = GetHbLazyTensor(self).getTensorUniqueId();
    StrideParams* params_ptr = context->viewContext.GetViewTableEntry(id);
    if (params_ptr != nullptr) {
      // add strided insert at the cast output
      at::TensorOptions options = src.options().dtype(self.scalar_type());
      auto src_cast = empty_hpu_lazy(
          src.sizes(), options, src.suggest_memory_format(), false);

      auto hl_src_cast = GetHbLazyTensor(src_cast);
      ir::Value& out = hl_src_cast.CurrentIrValue();
      out.SetNode(
          node,
          hl_src_cast.GetDevice(),
          hl_src_cast.GetSizes(),
          hl_src_cast.dtype_optional());
      flush_op(src_cast);

      HbLazyTensorViews::HandleViewsD2D(src_cast, self);
    } else {
      auto hlresult = GetHbLazyTensor(self);
      ir::Value& out = hlresult.CurrentIrValue();
      out.SetNode(
          node,
          hlresult.GetDevice(),
          hlresult.GetSizes(),
          hlresult.dtype_optional());
      flush_op(self);
    }

    updateDstDependencies(hlresult, self);
  }

  return self;
}

Tensor permute_hpu_lazy_internal(const Tensor& self, IntArrayRef dims_in) {
  PT_LAZY_TRACE;
  habana_lazy::SyncAccThreadPool();
  auto dims_vec = dims_in.vec();
  for (unsigned i = 0; i < dims_in.size(); i++) {
    dims_vec[i] = at::maybe_wrap_dim(dims_in[i], self.dim(), true);
  }
  IntArrayRef dims_(dims_vec);

  std::vector<at::IValue> vector_of_inputs;

  vector_of_inputs = {self, dims_};

  using T = at::Tensor;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(const std::vector<at::IValue>& vector_of_inputs)
        : LazyOp<T>("hpu::permute", vector_of_inputs, {}, {}, -1) {}

   private:
    T get_result_overrideable() override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      auto dims = inputs[1].toIntList();
      std::vector<int64_t> new_sizes, new_strides;
      std::tie(new_sizes, new_strides) =
          PermuteOperator::compute_output_shape(self, dims.vec());
      auto result =
          empty_strided_hpu_lazy(new_sizes, new_strides, self.options(), false);
      return result;
    }
  };

  Kernel kernel{vector_of_inputs};
  return kernel.call();
}

Tensor as_strided_layout_hpu_lazy(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride) {
  int64_t dim_out_pos[] = {
      LayoutFormatDims::H,
      LayoutFormatDims::W,
      LayoutFormatDims::C,
      LayoutFormatDims::N};
  int64_t dim_out_pos_3d[] = {
      LayoutFormatWithDepthDims::D,
      LayoutFormatWithDepthDims::H,
      LayoutFormatWithDepthDims::W,
      LayoutFormatWithDepthDims::C,
      LayoutFormatWithDepthDims::N};
  IntArrayRef dims_ = dim_out_pos;
  if (self.dim() == 5)
    dims_ = dim_out_pos_3d;
  auto node = std::make_shared<ir::AsStridedLayout>(
      self, dims_, "hpu::as_strided_layout");
  auto result = empty_strided_hpu_lazy(size, stride, self.options(), false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  return result;
}

c10::optional<at::Tensor> handleWeightTensorLayout(const Tensor& src) {
  static std::vector<int> out_pos = {
      LayoutFormatDims::H,
      LayoutFormatDims::W,
      LayoutFormatDims::C,
      LayoutFormatDims::N};
  static std::vector<int> out_pos_5d = {
      LayoutFormatWithDepthDims::D,
      LayoutFormatWithDepthDims::H,
      LayoutFormatWithDepthDims::W,
      LayoutFormatWithDepthDims::C,
      LayoutFormatWithDepthDims::N};

  auto sizes = src.sizes().vec();
  auto is_5d_tensor = src.dim() == 5;
  auto hb_tensor = GetHbLazyTensor(src);
  auto tensor_data = hb_tensor.GetHbLazyTensorData();
  auto hl_tensor_data = habana_lazy::GetHbInternalTensorImpl(*tensor_data);

  // weights HWCK -> NCHW
  if ((hl_tensor_data->GetTensorLayout() == habana_lazy::LayoutFormat::kHWCK) &&
      (habana_lazy::exec::OptPassCfg::GetInstance()
           ->IsEnabledWeightPermutePass())) {
    std::vector<long int> swapped_sizes = {
        sizes[out_pos[0]],
        sizes[out_pos[1]],
        sizes[out_pos[2]],
        sizes[out_pos[3]]};
    std::vector<long int> swapped_sizes_5d = {
        sizes[out_pos_5d[0]],
        sizes[out_pos_5d[1]],
        sizes[out_pos_5d[2]],
        sizes[out_pos_5d[3]],
        sizes[out_pos_5d[4]]};
    if (is_5d_tensor) {
      auto new_strides =
          CalculateStrides5d(swapped_sizes_5d, c10::MemoryFormat::Contiguous);
      auto strided_tensor =
          // as_strided_hpu_lazy(src, swapped_sizes_5d, new_strides, 0);
          as_strided_layout_hpu_lazy(src, swapped_sizes_5d, new_strides);
      auto permute_tensor = permute_hpu_lazy_internal(
          strided_tensor,
          {LayoutFormatWithDepthDims::W,
           LayoutFormatWithDepthDims::H,
           LayoutFormatWithDepthDims::N,
           LayoutFormatWithDepthDims::C,
           LayoutFormatWithDepthDims::D});
      HbLazyTensor hb_tensor = GetHbLazyTensor(permute_tensor);
      tensor_data = hb_tensor.GetHbLazyTensorData();
    } else {
      auto new_strides =
          CalculateStrides(swapped_sizes, c10::MemoryFormat::Contiguous);
      auto strided_tensor =
          // as_strided_hpu_lazy(src, swapped_sizes, new_strides, 0);
          as_strided_layout_hpu_lazy(src, swapped_sizes, new_strides);
      auto permute_tensor = permute_hpu_lazy_internal(
          strided_tensor,
          {LayoutFormatDims::W,
           LayoutFormatDims::H,
           LayoutFormatDims::N,
           LayoutFormatDims::C});
      HbLazyTensor hb_tensor = GetHbLazyTensor(permute_tensor);
      tensor_data = hb_tensor.GetHbLazyTensorData();
    }
  }
  return tensor_data;
}

void validateHbTensorData(HbLazyTensor& hb_tensor) {
  auto hb_tensor_data = hb_tensor.GetHbLazyTensorData();
  if (!hb_tensor_data) {
    TORCH_CHECK(
        false, "Habana Lazy: no storage tensor attached to lazy tensor");
  }
  if (!hb_tensor_data.value().has_storage()) {
    TORCH_CHECK(false, "Habana Lazy: lazy tensor doesn't has a storage");
  }
}

Tensor& copy_hpu_lazy_D2H(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;

  // This situation should not occur
  // Throwing an exception here for now to catch any cases that arise
  TORCH_CHECK(
      IsHbLazyTensor(src),
      "Habana Lazy : trying to copy back a tensor which does not have a lazy tensor");

  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
  context->JoinPendingLaunchThread();

  // Remove this SBS check, as of now prepare sbs inputs on calls .to operator
  // and on inplace ops that triggers this mark step, which leads to graph
  // evaluation and we end up losing input tensor.
  if (GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_DISABLED) {
    auto hl_t = GetHbLazyTensor(src);
    if (hl_t.CurrentIrValue() && !hl_t.CurrentIrValue().IsHpuInputNode()) {
      PT_LAZY_DEBUG("Triggering mark_step before D2H copy");
      HbLazyTensor::StepMarker({});
    }
  }
  // handle views
  auto _src = HbLazyTensorViews::HandleViewsD2H(src);
  auto hb_tensor = GetHbLazyTensor(_src);
  validateHbTensorData(hb_tensor);

  // If _src is a lazy tensor make sure the execution till the point of _src
  // getting flled has finished before we start copying
  auto tensor_data = handleWeightTensorLayout(_src);

  TORCH_CHECK(
      tensor_data, "Trying to copy from lazy tensor with no backend memory");
  auto type = hb_tensor.getTensorOriginalType();
  // This path is disabled for now, when we return back from Habana to
  // CPU we can check if the original tensor was long/double , if soe we
  // can upscale it and send it back. For now we just send the 32bit
  // tensor that Habana holds

  // suggest_memory_format() uses strides and sizes to determine the memory
  // format. Depending on strided_view's stride params, the self (and src)
  // memory format can be incorrecly mapped to ch last or ch last 3d. Refer:
  // LazyBasicKernelTest.noncontiguous. Use backend tensors memory format to
  // correctly identify the memory format
  self = self.contiguous(tensor_data.value().suggest_memory_format());
  if (type != typeMetaToScalarType(_src.dtype())) {
    // If we need to upscale the CPU tensor using the .to for now
    // It rebinds the self reference to the new tensor
    // We need to check the memory deletion of the original tensor created
    // by PT
    PT_LAZY_DEBUG(
        "WARNING: We are hitting a case in H2D where the PyTorch tensor original data types mismatch.");
    self = self.to(_src.dtype());
    self = copy_hpu_(self, tensor_data.value(), non_blocking);
    self = self.to(type);
  } else {
    self = copy_hpu_(self, tensor_data.value(), non_blocking);
  }
  // No need to CreateHbLazyTensor for self as it is on CPU
  habana_lazy::PermuteTensors::handlePermutedTensor(_src, self, non_blocking);
  return self;
}

void calculate_size_stride_cl(
    int64_t dim,
    std::vector<int64_t>& size,
    std::vector<int64_t>& stride,
    std::vector<int64_t>& permute_dims) {
  std::iota(permute_dims.begin(), permute_dims.end(), -1);
  // prepare the permute params to channels first.
  permute_dims[0] = 0;
  permute_dims[1] = dim - 1;
  auto temp = size[1];
  for (int i = 1; i < dim - 1; i++) {
    size[i] = size[i + 1];
  }
  size[dim - 1] = temp;
  habana_helpers::recalc_strides(stride, size);
}

static Tensor permute_hpu_lazy_cl(const Tensor& self, IntArrayRef dims_in) {
  PT_LAZY_TRACE;
  auto dims_vec = dims_in.vec();
  for (unsigned i = 0; i < dims_in.size(); i++) {
    dims_vec[i] = at::maybe_wrap_dim(dims_in[i], self.dim(), true);
  }
  IntArrayRef dims_(dims_vec);

  std::vector<at::IValue> vector_of_inputs;

  vector_of_inputs = {self, dims_};

  using T = at::Tensor;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(const std::vector<at::IValue>& vector_of_inputs)
        : LazyOp<T>("aten::permute", vector_of_inputs, {}, {}, -1) {}

   private:
    T get_result_overrideable() override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      auto dims = inputs[1].toIntList();
      std::vector<int64_t> new_sizes, new_strides;
      std::tie(new_sizes, new_strides) =
          PermuteOperator::compute_output_shape(self, dims.vec());
      auto result =
          empty_strided_hpu_lazy(new_sizes, new_strides, self.options(), false);
      return result;
    }
  };

  Kernel kernel{vector_of_inputs};
  return kernel.call();
}

Tensor& copy_hpu_lazy_H2D(Tensor& self, const Tensor& src_, bool non_blocking) {
  PT_LAZY_TRACE;
  bool processed = false;
  auto src = src_.contiguous(src_.suggest_memory_format());
  InitSizesAndStrides(
      self,
      c10::nullopt,
      self.sizes(),
      c10::nullopt,
      self.suggest_memory_format());
  auto exec_mode = habana_lazy_executor.getExecutionMode();
  if (exec_mode != kLOWERING) {
    auto self_hb_tensor = GetOrCreateHbLazyTensor(self, self.device());
    // WE need to add storage if it wasnt created
    // right now as soon as we do a H2D transfer, we create memory and mark
    // executed
    auto isStorageAttached = self_hb_tensor.isStorageAttached();
    if (!isStorageAttached) {
      if (!habana_helpers::is_supported_type(src.scalar_type()) or
          !habana_helpers::is_supported_type(self.scalar_type())) {
        PT_LAZY_WARN(
            "Falling back to CPU - Unsupported src or dst types in H2D copy: src: ",
            src.scalar_type(),
            ", dst: ",
            self.scalar_type());
        auto fb_src = at::empty_like(src);
        return fb_src.copy_(src);
      }
      c10 ::Allocator* allocator;
      allocator = habana::getHABANADeviceAllocator();
      int64_t nelements = multiply_integers(self.sizes());
      int elem_size = self.dtype().itemsize();
      int64_t size_bytes = nelements * elem_size;
      auto storage_impl = c10::make_intrusive<StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size_bytes,
          allocator->allocate(nelements * elem_size),
          allocator,
          /*resizeable=*/true);
      Tensor at_internal_tensor = AtenInternalHbTensor(
          std::move(storage_impl),
          self.dtype(),
          c10::nullopt,
          src.sizes(),
          c10::nullopt,
          src.suggest_memory_format());
      // Setup the tensor sizes & strides for tensor with dim = 4, else for
      // now assuming contiguous
      self_hb_tensor.SetTensorData(at_internal_tensor);
    }
  }
  auto new_tensor = preProcessIfLongorDouble(src, self, processed);

  // Get the internal tensor for copy kernel
  // First get the lazy tensor
  auto self_hb_tensor = GetOrCreateHbLazyTensor(self, self.device());
  auto context =
      habana_lazy_executor.getDeviceExecutionContext(self.device().index());
  if (self_hb_tensor.IsExecutionInProgress()) {
    context->JoinPendingLaunchThread();
  }

  if (self_hb_tensor.CurrentIrValue() &&
      !self_hb_tensor.CurrentIrValue().IsHpuInputNode()) {
    PT_LAZY_DEBUG("Triggering mark_step before H2D copy");
    HbLazyTensor::StepMarker({});
  }

  // Set the tensor as input and mark as input
  setTensorAsInputNode(self_hb_tensor);

  context->MarkTensorStatus(
      self_hb_tensor.getDataPtr(), LazyTensorExecutionStatus::kINPUT);
  auto id = self_hb_tensor.getTensorUniqueId();
  StrideParams* params_ptr = context->viewContext.GetViewTableEntry(id);

  // We need to mark this tensor as executed
  // As this will be an input coming from host side, its doesnt need further
  // execution and is ready for consumption as input
  auto self_hb_tensor_data = self_hb_tensor.GetHbLazyTensorData();
  // This is the internal tensor, it isn't a lazy tensor
  auto self_internal_tesor = self_hb_tensor_data.value();

  HABANA_ASSERT(!TryGetHbLazyTensor(self_internal_tesor));
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    auto hb_impl = habana_lazy::GetHbInternalTensorImpl(self_internal_tesor);
    auto synapse_permute = hb_impl->GetMemoryPermutation();
    if (synapse_permute.size() != 0) {
      PT_LAYOUTS_DEBUG(
          "clearing memory permute, id ",
          self_hb_tensor.getTensorUniqueId(),
          " permute ",
          VecToString(synapse_permute))
      hb_impl->SetMemoryPermutation({});
    }
  }

  // self may have been resized, so re-set its size and strides
  self_internal_tesor.unsafeGetTensorImpl()->set_sizes_and_strides(
      self.sizes(), self.strides());
  if (processed) {
    auto internal_tensor_from_copy =
        copy_hpu_(self_internal_tesor, new_tensor, non_blocking);
    // We should get back the same internal tensor passed to copy
    HABANA_ASSERT(
        self_internal_tesor.storage().data_ptr() ==
        internal_tensor_from_copy.storage().data_ptr());
  } else {
    auto internal_tensor_from_copy =
        copy_hpu_(self_internal_tesor, src, non_blocking);
    // We should get back the same internal tensor passed to copy
    HABANA_ASSERT(
        self_internal_tesor.storage().data_ptr() ==
        internal_tensor_from_copy.storage().data_ptr());
  }

  if ((self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast ||
       self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d) &&
      (self.dim() == 4 || self.dim() == 5)) {
    auto dim = self.dim();
    auto size = self.sizes().vec();
    auto stride = self.strides().vec();
    std::vector<int64_t> permute_dims(dim);
    calculate_size_stride_cl(dim, size, stride, permute_dims);

    self.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
    self_internal_tesor.unsafeGetTensorImpl()->set_sizes_and_strides(
        size, stride);
    self = permute_hpu_lazy_cl(self, permute_dims);
  }

  // TODO : Handle view table update of channels_last tensor
  if (params_ptr != nullptr) {
    strided_insert_hpu_lazy(self, self, false);
  }

  // Return the self tensor, as copy_hpu_ doesn't create a new tensor and
  // returns the dst
  flush_op(self);
  return self;
}

Tensor& copy_hpu_lazy_(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;
  TORCH_CHECK(self.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  const auto src_device = src.device().type();
  const auto dst_device = self.device().type();

  bool is_d2d_copy = false;
  if (src_device == c10::DeviceType::HPU &&
      dst_device == c10::DeviceType::HPU) {
    is_d2d_copy = true;
  }

  // If it isnt a device to device copy, we are transferring data to and
  // from CPU. This becomes an execution step point and we need to flush
  // graph execution NOW to generate tensor data where required as we are in
  // lazy mode. Otherwise we have to add the copy induced nodes(like cast)
  // to lazy graph for execution later
  if (!is_d2d_copy) {
    if (src_device == c10::DeviceType::CPU) {
      self = copy_hpu_lazy_H2D(self, src, non_blocking);
    } else if (src_device == c10::DeviceType::HPU) {
      self = copy_hpu_lazy_D2H(self, src, non_blocking);
    }
  } else {
    self = copy_hpu_lazy_D2D(self, src, non_blocking);
  }

  return self;
}

// This API should be called from lowering mode only
// It is used to create th backend tensor for as_strided
Tensor empty_as_strided_lazy(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  PT_LAZY_TRACE;
  auto storage_impl = self.unsafeGetTensorImpl();
  Tensor at_internal_tensor = AtenInternalHbTensor(
      c10::Storage(storage_impl->storage()),
      self.dtype(),
      c10::nullopt,
      size,
      stride,
      c10::nullopt);
  if (storage_offset) {
    at_internal_tensor.unsafeGetTensorImpl()->set_storage_offset(
        storage_offset.value());
  }

  auto hb_at_internal_self = habana_lazy::GetHbInternalTensorImpl(self);
  auto hb_at_internal_tensor =
      habana_lazy::GetHbInternalTensorImpl(at_internal_tensor);
  auto layout_format = hb_at_internal_self->GetTensorLayout();
  hb_at_internal_tensor->SetTensorLayout(layout_format);

  return at_internal_tensor;
}

ir::NodePtr create_as_strided_node(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset,
    bool is_out) {
  ir::NodePtr node = nullptr;

  auto offset = storage_offset.value_or(self.storage_offset());
  auto mf = self.suggest_memory_format();

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    std::string node_str =
        (is_out) ? "hpu::strided_view_out_ds" : "hpu::strided_view_ds";

    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
      node_str = ((mf == c10::MemoryFormat::ChannelsLast) ||
                  (mf == c10::MemoryFormat::ChannelsLast3d))
          ? "hpu::strided_view_cl_ds"
          : "hpu::strided_view_ds";
    }
    PT_DYNAMIC_SHAPE_DEBUG(
        "Strided view Real size = ",
        self.sizes().vec(),
        " recieved sizes = ",
        size.vec(),
        " strides = ",
        stride.vec(),
        " offset = ",
        offset);
    auto out_size_st = empty_hpu_lazy(
        size,
        self.options(),
        c10::MemoryFormat::Contiguous,
        false,
        SHAPE_TENSOR);
    std::vector<int64_t> offset_vec = {offset};
    IntArrayRef offset_ref(offset_vec.data(), offset_vec.size());
    auto offset_st = empty_hpu_lazy(
        offset_ref,
        self.options(),
        c10::MemoryFormat::Contiguous,
        false,
        SHAPE_TENSOR);
    if (self.sizes().size() != stride.size()) {
      if (node_str == "hpu::strided_view_out_ds") {
        node_str = "hpu::strided_view_out_orig_ds";
      } else {
        node_str = "hpu::strided_view_orig_ds";
      }
      auto stride_st = empty_hpu_lazy(
          stride,
          self.options(),
          c10::MemoryFormat::Contiguous,
          false,
          SHAPE_TENSOR);
      node = std::make_shared<ir::StridedView>(
          self, out_size_st, stride_st, offset_st, node_str);
    } else {
      auto lazy_ten = GetHbLazyTensor(out_size_st);
      auto tensor_size_st = lazy_ten.CurrentTensorAttached().value();
      auto impl_size_st = habana_lazy::GetHbInternalTensorImpl(tensor_size_st);
      HABANA_ASSERT(impl_size_st, "impl_size_st is invalid");

      std::vector<int64_t> stride_ratios;
      auto self_strides = self.strides().vec();
      auto stride_sizes = stride.vec();
      auto len = stride_sizes.size();
      for (uint64_t i = 0; i < len; i++) {
        stride_ratios.push_back(stride_sizes[i] / self_strides[i]);
      }
      impl_size_st->get_shape_struct().set_strides_tensor_shape(stride_sizes);
      impl_size_st->get_shape_struct().set_stride_ratio(stride_ratios);
      PT_DYNAMIC_SHAPE_DEBUG(
          "Setting stride ratio = ", stride_ratios, " offset = ", offset);

      node = std::make_shared<ir::StridedView>(
          self, out_size_st, offset_st, node_str);
    }
  } else {
    std::string node_str =
        (is_out) ? "hpu::strided_view_out" : "hpu::strided_view";

    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
      node_str = ((mf == c10::MemoryFormat::ChannelsLast) ||
                  (mf == c10::MemoryFormat::ChannelsLast3d))
          ? "hpu::strided_view_cl"
          : "hpu::strided_view";
    }
    node =
        std::make_shared<ir::StridedView>(self, size, stride, offset, node_str);
  }
  return node;
}

// THis kernel has two paths, lowering and lazy
// During lazy we set up the as strided tensor meta data
// when we get a call back from lowering, we attache the tensor from same memory
// as source
Tensor as_strided_hpu_lazy(
    const Tensor& self,
    IntArrayRef size_in,
    IntArrayRef stride_in,
    c10::optional<int64_t> storage_offset) {
  PT_LAZY_TRACE;
  habana_lazy::SyncAccThreadPool();

  // lazy within lazy. as strided node is not here. Only the view table update
  // happens here
  auto storage_offset_val = storage_offset.value_or(self.storage_offset());

  auto out = HbLazyTensorViews::add_strided_view_node(
      self,
      size_in,
      stride_in,
      storage_offset_val,
      true /*is_update_view*/,
      c10::nullopt);
  if (habana_lazy_executor.getExecutionMode() != kLOWERING) {
    flush_op(out);
  }
  return out;
};

const Tensor& as_strided_hpu_lazy_(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  // We only support contiguous chunks of data to be taken as strided,
  // as Device doesnt support strided tensors we dont support that case
  ir::NodePtr node = create_as_strided_node(self, size, stride, storage_offset);

  if (node != nullptr) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

    auto hb_result = GetHbLazyTensor(self);
    // update of lazy tensor size is required for permute pass to see output
    // with updated shape
    hb_result.setTensorSize(size.vec());
    ir::Value& out = hb_result.CurrentIrValue();
    out.SetNode(
        node,
        hb_result.GetDevice(),
        hb_result.GetSizes(),
        hb_result.dtype_optional());

    auto context = habana_lazy_executor.getDeviceExecutionContext();
    context->MarkTensorStatus(
        hb_result.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
    flush_op(self);
    return self;
  } else {
    TORCH_CHECK(
        0,
        "as_strided_ called with strides creating non-contiguous output tensor not supported");
  }
};

void AddMemcpy(const Tensor& src, Tensor& dst) {
  auto hl_dst = GetOrCreateHbLazyTensor(dst);
  auto hl_src = GetHbLazyTensor(src);
  // add control edge to avoid GC error " writing to already
  // registered graph output"

  // Add a control edge for habana_d2d_memcpy_other second input
  // as it may cause wrong order of execution, as shown below -
  //   z = add(x, i)
  //   x' = habana_d2d_memcpy_other(y, x)
  // Here, x is an input, but habana_d2d_memcpy_other actually updates
  // x and hence should come after add with a control edge
  updateDstDependencies(hl_dst, dst, true);

  auto copy_node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::habana_d2d_memcpy_other"),
      {hl_src.GetIrValue(), hl_dst.GetIrValue()});

  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      dst.device().index());
  context->RegisterTensor(hl_dst.getDataPtr());
  habana_lazy::ir::Value& out = hl_dst.CurrentIrValue();
  out.SetNode(
      copy_node,
      hl_dst.GetDevice(),
      hl_dst.GetSizes(),
      hl_dst.dtype_optional());
  std::vector<at::Tensor> input_pt_vec;
  input_pt_vec.push_back(src);
  input_pt_vec.push_back(dst);
  copy_node->AddInputPtTensors(input_pt_vec);
  flush_op(dst);
}

Tensor& set_hpu_lazy_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  PT_LAZY_TRACE
  // TODO Handle stride
  static_cast<void>(stride);

  auto lazy_ten = GetHbLazyTensor(self);
  auto impl = lazy_ten.getAttachedTensorImpl();
  HABANA_ASSERT(impl, "impl is invalid");
  impl->set_storage_keep_dtype(std::move(source));
  impl->set_storage_offset(storage_offset);
  self.resize_(size, self.suggest_memory_format());

  return self;
}

Tensor view_hpu_lazy(const Tensor& self_, IntArrayRef size) {
  PT_LAZY_TRACE;

  auto self = self_;

  auto hl_self = GetHbLazyTensor(self);

  // multilevel view optimization
  // v1 = view(a, out_size1)
  // v2 = view(v1, out_size2)
  // The above sequence can be compressed to v2 = view(a, out_size2)
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto self_id = hl_self.getTensorUniqueId();
  {
    LOCK_VIEW_TABLE_MUTEX(context->viewContext);
    StrideParams* params_ptr = context->viewContext.GetViewTableEntry(self_id);
    if (params_ptr != nullptr) {
      if (params_ptr->optype == kStridedOpView) {
        self = params_ptr->parent;
        PT_VIEWTABLE_DEBUG("invoked multilevel view optimization");
      }
    }
  }
  auto inferred_size = habana_helpers::infer_size(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto stride_value = *stride;

  auto out = as_strided_hpu_lazy(
      self, inferred_size, stride_value, self.storage_offset());
  auto hb_result = GetHbLazyTensor(out);
  {
    LOCK_VIEW_TABLE_MUTEX(context->viewContext);
    auto strided_param = HbLazyTensorViews::getViewTableParams(hb_result);
    // There could be some cases where slice/select/etc followed by view, in
    // those cases use as_strided instead of using the ViewOP.
    if (is_fallback_original_op(self, out)) {
      strided_param->optype = kStridedOpView;

      PT_VIEWTABLE_DEBUG(
          "view fallback- tensor id: ",
          hl_self.getTensorUniqueId(),
          "size ",
          size);
    }
  }
  return out;
}

void add_tensor_hpu_lazy_parallel_impl(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  PT_LAZY_TRACE;

  auto alpha_double = alpha.toDouble();
  if (alpha_double != 1.0) {
    at::Tensor alpha_tensor =
        get_tensor_for_scalar(alpha_double, other.options());

    auto hl_alpha = GetOrCreateHbLazyTensor(alpha_tensor, c10::kHPU);
    auto mul_out = torch::mul(other, alpha_tensor);
    if (other.unsafeGetTensorImpl()->is_wrapped_number()) {
      // The operation has been split into intermediate multiply and then again
      // add op tensor produced by this split resulted in inappropriate type
      // deduction of whole add operation. alpha is always scalar, when also
      // other is scalar then marking intermediate as wrapped number is also
      // necessary to further proper deduction
      mul_out.unsafeGetTensorImpl()->set_wrapped_number(true);
    }
    add_tensor_hpu_lazy_parallel_impl(self, mul_out, 1.0, out);
  } else {
    LazyBinaryOp<at::Tensor> k{
        "aten::add",
        {self, other, alpha},
        false,
        true,
        {},
        {BinaryOperator::compute_output_shape(self, other)},
        -1};
    k.call(out);
  }
}

Tensor add_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;

  LazyBinaryOp<at::Tensor> k{
      "aten::add",
      {self, other, alpha},
      false,
      true,
      {},
      {BinaryOperator::compute_output_shape(self, other)},
      -1};

  auto out = k.get_result();

  auto op_func = [self, other, alpha, out]() mutable {
    add_tensor_hpu_lazy_parallel_impl(self, other, alpha, out);
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(add, op_func, out);
}

Tensor add_scalar_hpu_lazy(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> op{
      "aten::add", {self, other, alpha}, {}, {self.sizes().vec()}};
  RUN_MAYBE_WITH_ACC_THREAD(add, op)
}

Tensor& add_scalar_hpu_lazy_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  auto other_tensor = get_tensor_for_scalar(other.toDouble(), self.options());

  return add_tensor_hpu_lazy_(self, other_tensor, alpha);
}

void add_tensor_hpu_lazy_inplace_parallel_impl(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  auto alpha_double = alpha.toDouble();
  if (alpha_double != 1.0) {
    at::Tensor alpha_tensor =
        get_tensor_for_scalar(alpha_double, other.options());

    auto hl_alpha = GetOrCreateHbLazyTensor(alpha_tensor, c10::kHPU);
    auto mul_out = torch::mul(other, alpha_tensor);
    add_tensor_hpu_lazy_inplace_parallel_impl(self, mul_out, 1.0);
  } else {
    LazyBinaryOp<Tensor&> op("aten::add_", {self, other, alpha}, false, true);
    op.call(self);
  }
}

Tensor& add_tensor_hpu_lazy_(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;

  if (habana_lazy::IsAccThreadEnabled()) {
    // try to construct DTypeHelper to make sure,
    // that type promotion does not throw due to incompatible dtypes
    at::IValue ivalue(self);
    auto output = c10::make_optional<const at::IValue*>(&ivalue);
    auto dtype_helper =
        habana_helpers::DTypeHelper::binary_op_with_type_promotion(
            {self, other}, output, true);
  }

  auto op_func = [self, other, alpha]() mutable {
    add_tensor_hpu_lazy_inplace_parallel_impl(self, other, alpha);
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(add_, op_func, self);
}

Tensor& mul_out_hpu_lazy(const Tensor& self, const Tensor& other, Tensor& out) {
  PT_LAZY_TRACE;
  // 8x all reduce optimization to avoid out variant that requires tensor with
  // storage. //TODO enhance lazy op framework to convert out variant to out of
  // place variant
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
  auto id = GetHbLazyTensor(out).getTensorUniqueId();
  {
    LOCK_VIEW_TABLE_MUTEX(context->viewContext);
    StrideParams* params_ptr = context->viewContext.GetViewTableEntry(id);
    if (params_ptr != nullptr) {
      auto orig_out = out;
      auto temp = torch::mul(self, other);
      Tensor temp_cast = temp;
      if (temp.scalar_type() != orig_out.scalar_type()) {
        // Cast temp tensor to orig_out tensor data type
        LazyOp<Tensor> k_{
            "hpu::cast",
            {temp, orig_out.scalar_type()},
            {},
            {temp.sizes().vec()}};
        temp_cast = k_.call();
      }
      strided_insert_hpu_lazy(orig_out, temp_cast);
    } else {
      std::vector<at::Tensor> metatens_tensors = {self, other, out};
      auto metatens = habana::GetMetaTensorList(metatens_tensors);
      at::TensorList metavar =
          at::mul_outf(metatens[0], metatens[1], metatens[2]);
      LazyBinaryOp<at::Tensor&> hpu_op{
          "aten::mul", {self, other, out}, true, true, metavar};
      return hpu_op.call(out);
    }
  }

  return out;
}

Tensor div_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  // The auto code gen way of implementing div with rounding mode is
  // more comprehensive. Hence use this op without specific mode
  // to realize normal div

  c10::optional<c10::string_view> mode = c10::nullopt;
  return torch::div(self, other, mode);
}
Tensor& div_tensor_hpu_lazy_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  PT_LAZY_TRACE;
  c10::optional<c10::string_view> mode = c10::nullopt;
  return torch::div_outf(self, other, mode, out);
}

Tensor& div_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  c10::optional<c10::string_view> mode = c10::nullopt;
  return self.div_(other, mode);
}

Tensor div_scalar_hpu_lazy(const Tensor& self, const Scalar& other) {
  PT_LAZY_TRACE;

  auto other_tensor = get_tensor_for_scalar(other.toDouble(), self.options());
  c10::optional<c10::string_view> mode = c10::nullopt;
  return torch::div(self, other_tensor, mode);
}

Tensor& div_scalar_hpu_lazy_(Tensor& self, const Scalar& other) {
  PT_LAZY_TRACE;
  c10::optional<c10::string_view> mode = c10::nullopt;
  return self.div_(other, mode);
}

Tensor permute_wt_hpu(const Tensor& self) {
  at::Tensor result = self;
  if (habana_lazy::exec::OptPassCfg::GetInstance()
          ->IsEnabledWeightPermutePass() &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1)) {
    if (self.dim() == 4 || self.dim() == 5) {
      auto hb_tensor = GetOrCreateHbLazyTensor(self, self.device());
      auto layout_format = hb_tensor.GetTensorLayout();

      int64_t dim_out_pos[] = {
          LayoutFormatDims::H,
          LayoutFormatDims::W,
          LayoutFormatDims::C,
          LayoutFormatDims::N};
      int64_t dim_out_pos_3d[] = {
          LayoutFormatWithDepthDims::D,
          LayoutFormatWithDepthDims::H,
          LayoutFormatWithDepthDims::W,
          LayoutFormatWithDepthDims::C,
          LayoutFormatWithDepthDims::N};
      IntArrayRef dims_ = dim_out_pos;
      if (self.dim() == 5)
        dims_ = dim_out_pos_3d;

      std::string op_name;

      if (layout_format != habana_lazy::LayoutFormat::kHWCK) {
        op_name = "hpu::permute_weight";
      } else {
        op_name = "hpu::permuted_weight_restride";
      }

      hb_tensor.SetTensorLayout(habana_lazy::LayoutFormat::kHWCK);
      std::vector<at::IValue> vector_of_inputs;
      vector_of_inputs = {self, dims_};

      using T = at::Tensor;
      class Kernel : public LazyOp<T> {
       public:
        Kernel(
            const std::string& op_name,
            const std::vector<at::IValue>& vector_of_inputs)
            : LazyOp<T>(op_name, vector_of_inputs, {}, {}, -1) {}

       private:
        T get_result_overrideable() override {
          auto inputs = get_inputs();
          auto self = inputs[0].toTensor();
          return empty_strided_hpu_lazy(
              self.sizes(), self.strides(), self.options(), false);
        }
      };
      Kernel kernel{op_name, vector_of_inputs};
      return kernel.call();
    }
  }
  return result;
}

Tensor convolution_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  PT_LAZY_TRACE;
  const auto& bias = bias_opt.value_or(Tensor());
  Tensor weight_hpu = weight;
  if (weight.device().type() == c10::DeviceType::CPU &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING))
    weight_hpu = weight.to(c10::kHPU, true);

  if (habana_lazy::exec::OptPassCfg::GetInstance()
          ->IsEnabledWeightPermutePass()) {
    weight_hpu = weight.to(c10::kHPU, true);
    HbLazyTensor src_hb_tensor =
        GetOrCreateHbLazyTensor(weight_hpu, weight_hpu.device());
    if (src_hb_tensor.isStorageAttached()) {
      auto at_internal_tensor = *(src_hb_tensor.GetHbLazyTensorData());
      if (at_internal_tensor.has_storage()) {
        auto hb_tensor =
            habana_lazy::GetHbInternalTensorImpl(at_internal_tensor);
        hb_tensor->SetTensorLayout(habana_lazy::LayoutFormat::kHWCK);
      }
    }
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE)) {
    habana_lazy::PermuteTensors::permuteWeight(weight_hpu);
  }

  auto weight_hwck = permute_wt_hpu(weight_hpu);

  if (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE)) {
    auto hb_tensor = GetHbLazyTensor(weight_hwck);
    if (hb_tensor.isStorageAttached()) {
      auto at_internal_tensor = *(hb_tensor.GetHbLazyTensorData());
      if (at_internal_tensor.has_storage()) {
        auto internal_tensor =
            habana_lazy::GetHbInternalTensorImpl(at_internal_tensor);
        internal_tensor->SetConstTensor(true);
      }
    }
    if (bias.defined()) {
      auto hb_tensor = GetHbLazyTensor(bias);
      if (hb_tensor.isStorageAttached()) {
        auto at_internal_tensor = *(hb_tensor.GetHbLazyTensorData());
        if (at_internal_tensor.has_storage()) {
          auto internal_tensor =
              habana_lazy::GetHbInternalTensorImpl(at_internal_tensor);
          internal_tensor->SetConstTensor(true);
        }
      }
    }
  }

  bool is_weight_hwck = (habana_lazy::exec::OptPassCfg::GetInstance()
                             ->IsEnabledWeightPermutePass())
      ? false
      : true;
  if (weight_hwck.device().type() == c10::DeviceType::CPU &&
      (!habana_lazy::exec::OptPassCfg::GetInstance()
            ->IsEnabledWeightPermutePass()) &&
      !GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    auto is_5d_layout = weight_hwck.dim() == 5;
    c10::MemoryFormat memory_format = is_5d_layout
        ? c10::MemoryFormat::ChannelsLast3d
        : c10::MemoryFormat::ChannelsLast;
    std::array<int64_t, 4> swapped_dims_4d = {
        LayoutFormatDims::H,
        LayoutFormatDims::W,
        LayoutFormatDims::C,
        LayoutFormatDims::N};
    std::array<int64_t, 5> swapped_dims_5d = {
        LayoutFormatWithDepthDims::D,
        LayoutFormatWithDepthDims::H,
        LayoutFormatWithDepthDims::W,
        LayoutFormatWithDepthDims::C,
        LayoutFormatWithDepthDims::N};
    IntArrayRef dims_ = swapped_dims_4d;
    if (is_5d_layout)
      dims_ = swapped_dims_5d;
    std::vector<int64_t> strides(weight_hwck.sizes().size());
    weight_hwck = weight_hwck.permute(dims_).contiguous(memory_format);
    habana_helpers::recalc_strides(strides, weight_hwck.sizes().vec());
    IntArrayRef new_strides = strides;
    weight_hwck.unsafeGetTensorImpl()->set_sizes_and_strides(
        weight_hwck.sizes(), new_strides);
  }
  LazyOp<at::Tensor> k(
      "aten::convolution_overrideable",
      {input,
       weight_hwck,
       bias,
       stride,
       padding,
       dilation,
       transposed,
       output_padding,
       groups},
      {3, 4, 5, 6, 7, 8},
      {ConvOperator::compute_output_shape(
          input.sizes().vec(),
          weight_hwck.sizes().vec(),
          padding.vec(),
          stride.vec(),
          dilation.vec(),
          false,
          transposed,
          c10::MemoryFormat::Contiguous,
          false,
          is_weight_hwck,
          groups)},
      0);
  return k.call();
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  PT_LAZY_TRACE;
  auto weight_hwck = permute_wt_hpu(weight);
  // Construct using LazyOp templated with class ir::Convolution
  std::vector<bool> output_mask_vec(output_mask.begin(), output_mask.end());
  ir::NodePtr node = std::make_shared<ir::Convolution>(
      grad_output,
      input,
      weight_hwck,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      output_mask_vec);

  using T = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
  using U = ir::Convolution;
  class Kernel : public LazyOp<T, U> {
   public:
    Kernel(
        ir::NodePtr node,
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor weight,
        IntArrayRef stride,
        IntArrayRef padding,
        IntArrayRef dilation,
        bool transposed,
        IntArrayRef output_padding,
        int64_t groups,
        std::array<bool, 3> output_mask)
        : LazyOp<T, U>(
              std::move(node),
              {grad_output,
               input,
               weight,
               stride,
               padding,
               dilation,
               transposed,
               output_padding,
               groups,
               output_mask},
              {3, 4, 5, 6, 7, 8, 9},
              {},
              -1),
          grad_output{std::move(grad_output)},
          input{std::move(input)},
          weight{std::move(weight)} {}

   private:
    std::tuple<at::Tensor, at::Tensor, at::Tensor> get_result_overrideable()
        override {
      c10::MemoryFormat memory_format =
          habana_helpers::get_memory_format({&input, &grad_output, &weight});
      auto grad_input = empty_hpu_lazy(
          input.sizes(), grad_output.options(), memory_format, false);
      auto grad_weight = empty_hpu_lazy(
          weight.sizes(), grad_output.options(), memory_format, false);
      auto grad_bias = empty_hpu_lazy(
          {grad_output.size(1)}, grad_output.options(), memory_format, false);
      return {grad_input, grad_weight, grad_bias};
    }
    Tensor grad_output;
    Tensor input;
    Tensor weight;
  };

  Kernel k(
      node,
      grad_output,
      input,
      weight,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      output_mask);
  return k.call();
}

Tensor constant_pad_hpu_lazy(
    const Tensor& self,
    IntArrayRef pad,
    const Scalar& value) {
  PT_LAZY_TRACE;

  auto sizes = PadOperator::compute_output_shape(self, pad);
  auto out = empty_hpu_lazy(
      sizes, self.options(), self.suggest_memory_format(), false);

  std::vector<int64_t> pad_vec = pad.vec();
  auto func = [pad_vec = std::move(pad_vec), out, self, value]() mutable {
    IntArrayRef pad = pad_vec;
    std::vector<at::IValue> vector_of_inputs;
    std::string op_name;
    std::set<size_t> metadata_indices;
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
      // Keep IDST implementation also, but use H2D implementation by default
      bool isIDST =
          (GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_PAD_HOST_TENSOR) == false);
      if (isIDST) {
        op_name = "hpu::constant_pad_nd";
        std::vector<int64_t> pad_before(MAX_DIMENSIONS_NUM);
        std::vector<int64_t> pad_after(MAX_DIMENSIONS_NUM);

        for (unsigned int i = 0; i < pad.size() / 2; i++) {
          pad_before[MAX_DIMENSIONS_NUM - i - 1] = pad[2 * i];
          pad_after[MAX_DIMENSIONS_NUM - i - 1] = pad[2 * i + 1];
        }

        auto pad_before_tensor = empty_hpu_lazy(
            IntArrayRef(pad_before),
            self.options().dtype(c10::ScalarType::Int),
            self.suggest_memory_format(),
            false,
            INPUT_DESCRIBING_SHAPE_TENSOR);
        auto pad_after_tensor = empty_hpu_lazy(
            IntArrayRef(pad_after),
            self.options().dtype(c10::ScalarType::Int),
            self.suggest_memory_format(),
            false,
            INPUT_DESCRIBING_SHAPE_TENSOR);

        vector_of_inputs = {self, pad_before_tensor, pad_after_tensor, value};
      } else {
        op_name = "hpu::constant_pad_nd_ht";
        std::vector<uint32_t> pad_ht_vec(MAX_DIMENSIONS_NUM * 2, 0);
        // assuming that "pad" has a pair of pad values corresponding to each
        // dim that needs to be padded.
        for (unsigned int i = 0; i < pad.size() / 2; i++) {
          // Host tensor layout 1D - 10 elements: pad_before[0]...pad_before[4],
          // pad_after[0] ... pad_after[4] (for dimensionality IFM less then 5
          // some elements not in use)
          pad_ht_vec[i] = pad[2 * i];
          pad_ht_vec[MAX_DIMENSIONS_NUM + i] = pad[2 * i + 1];
        }

        auto pad_tensor = empty_hpu_lazy(
            pad_ht_vec.size(),
            self.options().dtype(c10::ScalarType::Int),
            self.suggest_memory_format(),
            false,
            HOST_TO_DEVICE_TENSOR);
        auto output_shape_tensor = empty_hpu_lazy(
            IntArrayRef(PadOperator::compute_output_shape(self, pad)),
            self.options().dtype(c10::ScalarType::Int),
            self.suggest_memory_format(),
            false,
            SHAPE_TENSOR);
        // Mark this front end shape tensor as it does not need synapse tensor
        auto hl_output_shape_tensor =
            GetOrCreateHbLazyTensor(output_shape_tensor, c10::kHPU);
        auto hl_output_shape_tensor_internal =
            hl_output_shape_tensor.CurrentTensorAttached().value();
        auto stImpl = habana_lazy::GetHbInternalTensorImpl(
            hl_output_shape_tensor_internal);
        if (stImpl) {
          stImpl->setH2DFrontEndShapeTensor();
        }
        auto hl_params_shape = GetOrCreateHbLazyTensor(pad_tensor, c10::kHPU);

        auto hl_param_internal =
            hl_params_shape.CurrentTensorAttached().value();
        habana_lazy::HbInternalTensorImpl* impl =
            habana_lazy::GetHbInternalTensorImpl(hl_param_internal);
        HABANA_ASSERT(impl);
        impl->set_host_data(
            pad_ht_vec.data(),
            pad_ht_vec.size(),
            sizeof(uint32_t),
            HostDataType::UINT32_T);
        vector_of_inputs = {self, pad_tensor, output_shape_tensor, value};
        metadata_indices = {3};
      }
    } else {
      op_name = "aten::constant_pad_nd";
      vector_of_inputs = {self, pad, value};
      metadata_indices = {1, 2};
    }
    LazyOp<at::Tensor> k{
        op_name,
        vector_of_inputs,
        metadata_indices,
        {PadOperator::compute_output_shape(self, pad)}};
    k.call(out);
  };
  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(constant_pad_nd, func, out)
}

Tensor embedding_hpu_lazy(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  PT_LAZY_TRACE;
  ir::NodePtr embedding_node = std::make_shared<ir::Embedding_forward>();

  // allocate Output storage
  auto size = indices.sizes().vec();

  if (indices.dim() == 1) {
    size = GatherOperator::compute_output_shape(weight, 0, indices);
  } else {
    // append size of last N-1 dimensions of weight (assuming its a Nd tensor)
    for (auto d : weight.sizes().slice(1)) {
      size.push_back(d);
    }
  }

  LazyOp<at::Tensor, ir::Embedding_forward> op(
      embedding_node,
      {weight, indices, padding_idx, scale_grad_by_freq, sparse},
      {size});
  auto out = op.get_result();

  auto func = [op = std::move(op),
               node = std::move(embedding_node),
               out,
               weight,
               indices,
               padding_idx,
               scale_grad_by_freq,
               sparse]() mutable {
    auto node_derived = std::dynamic_pointer_cast<ir::Embedding_forward>(node);
    node_derived->Init(
        weight, indices, padding_idx, scale_grad_by_freq, sparse);

    op.call(out);
  };
  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(embedding, func, out)
}

Tensor embedding_dense_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  PT_LAZY_TRACE;
  ir::NodePtr embedding_bwd_node = std::make_shared<ir::Embedding_backward>();
  std::vector<int64_t> sizes{num_weights, grad.size(-1)};

  LazyOp<at::Tensor, ir::Embedding_backward> op(
      embedding_bwd_node,
      {grad, indices, num_weights, padding_idx, scale_grad_by_freq},
      {sizes});

  auto out = op.get_result();

  auto func = [op = std::move(op),
               node = std::move(embedding_bwd_node),
               out,
               grad,
               indices,
               num_weights,
               padding_idx,
               scale_grad_by_freq]() mutable {
    auto node_derived = std::dynamic_pointer_cast<ir::Embedding_backward>(node);
    node_derived->Init(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);

    op.call(out);
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(embedding_dense_backward, func, out)
}
Tensor embedding_bag_sum_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_LAZY_TRACE;

  ir::NodePtr node = std::make_shared<ir::EmbeddingBagSum>();
  std::vector<int64_t> sizes{offsets.sizes()[0] - 1, input.size(1)};

  LazyOp<at::Tensor, ir::EmbeddingBagSum> op(
      node, {input, indices, offsets, valid_count, kernel_mode}, {sizes});
  auto out = op.get_result();

  auto func = [op = std::move(op),
               node = std::move(node),
               out,
               input,
               indices,
               offsets,
               valid_count,
               kernel_mode]() mutable {
    auto node_derived = std::dynamic_pointer_cast<ir::EmbeddingBagSum>(node);
    node_derived->Init(input, indices, offsets, valid_count, kernel_mode);

    op.call(out);
  };
  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(embedding_bag_sum, func, out)
}
Tensor embedding_bag_sum_fwd_hpu_lazy(
    const Tensor& input,
    const Tensor& indices_fwd,
    const Tensor& offsets_fwd,
    const Tensor& valid_count,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd,
    const Tensor& grad_weight) {
  PT_LAZY_TRACE;
  static_cast<void>(valid_count_bwd);
  std::vector<int64_t> sizes{offsets_fwd.numel() - 1, input.size(1)};
  LazyOp<at::Tensor> op{
      "aten::embedding_bag_sum_fwd",
      {input,
       indices_fwd,
       offsets_fwd,
       valid_count,
       indices_bwd,
       offsets_bwd,
       valid_count_bwd,
       grad_weight},
      {sizes}};

  RUN_MAYBE_WITH_ACC_THREAD(embedding_bag_sum_fwd, op)
}
Tensor& embedding_bag_sum_bwd_out_hpu_lazy(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> op{
      "aten::embedding_bag_sum_bwd",
      {out, input, indices_bwd, offsets_bwd, valid_count_bwd}};
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(embedding_bag_sum_bwd_out, op, out)
}
Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_lazy(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_LAZY_TRACE;

  ir::NodePtr node = std::make_shared<ir::EmbeddingBagSumBwd>();
  LazyOp<at::Tensor&, ir::EmbeddingBagSumBwd> op(
      node, {out, input, indices, offsets, valid_count, kernel_mode});

  auto func = [op = std::move(op),
               node = std::move(node),
               out,
               input,
               indices,
               offsets,
               valid_count,
               kernel_mode]() mutable {
    auto derived_node = std::dynamic_pointer_cast<ir::EmbeddingBagSumBwd>(node);
    derived_node->Init(out, input, indices, offsets, valid_count, kernel_mode);
    op.call(out);
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(embedding_bag_sum_bwd_out, func, out)
}

Tensor& fill_hpu_lazy_(Tensor& self, const Scalar& value) {
  PT_LAZY_TRACE;
  // This WA can be removed once GC fixes SW-70270
  // If self is a ZST then return it as it is since there is nothing to fill
  if (!self.numel() && (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2))
    return self;

  if (value.isBoolean()) {
    int bool_val = value.toBool();
    LazyOp<at::Tensor&> k{"aten::fill_", {self, bool_val}};
    RUN_INPLACE_MAYBE_WITH_ACC_THREAD(fill_, k, self)
  }

  LazyOp<at::Tensor&> k{"aten::fill_", {self, value}};
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(fill_, k, self)
}

Tensor& masked_fill_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  PT_LAZY_TRACE;
  // TPC doesn't support inplace where natively
  // Implement using out of place where followed by D2D copy
  // TODO revisit once strided mem copy feature is mature

  LazyOp<Tensor> where_op(
      "aten::where",
      {mask, value, self},
      {},
      {},
      2 /*output metadata is picked from self*/);

  Tensor where_out = where_op.call();
  // add a control edge as we add a loop using d2d copy back to self
  auto hl_self = GetOrCreateHbLazyTensor(self);
  // Adding memcpy to copy the output back to self as this is an inplace op
  AddMemcpy(where_out, self);
  return self;
}

Tensor& masked_fill_scalar_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    const Scalar& value) {
  PT_LAZY_TRACE;
  auto value_tensor = get_tensor_for_scalar(value.toDouble(), self.options());
  return masked_fill_hpu_lazy_(self, mask, value_tensor);
}

Tensor scatter_add_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k{"aten::scatter_add", {self, dim_, index, src}};
  RUN_MAYBE_WITH_ACC_THREAD(scatter_add, k)
}

// scatter_add is producing wrong value randomly
// https://jira.habana-labs.com/browse/SW-44742
Tensor& scatter_add_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_LAZY_TRACE;
  auto node =
      std::make_shared<habana_lazy::ir::ScatterAdd>(self, dim_, index, src);
  LazyOp<at::Tensor, ir::ScatterAdd> k{node, {self, dim_, index, src}};
  auto result = k.call();
  auto hl_self = GetOrCreateHbLazyTensor(self);
  // Create MemCopy operator to copy value into self
  AddMemcpy(result, self);
  return self;
}

Tensor& _index_put_impl_hpu_lazy_(
    Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const Tensor& value,
    bool accumulate,
    UNUSED const bool unsafe) {
  PT_LAZY_TRACE;
  // index backward is not supported on hpu, indices needs to be
  // bool, byte or long type for cpu fallback
  return index_put_hpu_lazy_(self, indices, value, accumulate);
}

Tensor slice_shape_tensor(const Tensor& shape_tensor) {
  std::vector<at::IValue> vector_of_inputs = {shape_tensor, 0, 1};
  auto end_shape = DimVector{1};

  using T = at::Tensor;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(const std::vector<at::IValue>& vector_of_inputs)
        : LazyOp<T>("aten::select", vector_of_inputs, {}, {}, -1) {}

   private:
    T get_result_overrideable() override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      auto end_shape = DimVector{1};
      return empty_hpu_lazy(
          end_shape,
          self.options().dtype(c10::ScalarType::Long),
          self.suggest_memory_format(),
          false);
    }
  };

  Kernel kernel{vector_of_inputs};
  return kernel.call();
}

Tensor nonzero_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  int elements = self.numel();
  at::TensorOptions hb_options = self.options();
  hb_options = hb_options.dtype(c10::ScalarType::Long);

  // Handle case for empty tensor where we return empty tensor with size
  if (elements == 0) {
    auto shape = DimVector{0, dimensions};
    auto output =
        empty_hpu_lazy(shape, hb_options, self.suggest_memory_format(), true);
    auto hl_output = GetHbLazyTensor(output);
    updateDstDependencies(hl_output, output);
    flush_op(output);
    return output;
  }

  using T = std::tuple<at::Tensor, at::Tensor>;
  struct NonZero : LazyOp<T> {
    explicit NonZero(
        const std::vector<at::IValue>& inputs,
        const std::set<size_t>& metadata_indices = {},
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor>>(
              "hpu::nonzero",
              inputs,
              metadata_indices,
              out_shapes,
              -1) {}

    std::tuple<at::Tensor, at::Tensor> get_result_overrideable() override {
      auto inputs = get_inputs();
      auto outputs = get_out_shapes();
      auto self = inputs[0].toTensor();
      auto where_tensor = empty_hpu_lazy(
          outputs[0],
          self.options().dtype(c10::ScalarType::Long),
          self.suggest_memory_format(),
          false);
      auto shape_tensor = empty_hpu_lazy(
          outputs[1],
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format(),
          false);
      return {where_tensor, shape_tensor};
    }
  };

  // Add nonzero node
  auto output_shape = NonZeroOperator::compute_output_shape(self);
  std::vector<int64_t> shape_tensor_shape{5};
  Tensor nz_shape_tensor;
  c10::optional<at::Tensor> nonzero_shape_tensor =
      c10::make_optional(nz_shape_tensor);
  NonZero k({self, c10::nullopt}, {}, {output_shape, shape_tensor_shape});
  // nonzero returns 2 output where and shape tensor
  auto result_nonzero = k.call();
  auto where_tensor = std::get<0>(result_nonzero);
  auto shape_tensor = std::get<1>(result_nonzero);

  // Select second element from shape tensor
  auto end_tensor = slice_shape_tensor(shape_tensor);
  PT_IRGRAPH_DEBUG("step marker due to non zero");
  // .item() internally triggers a mark_step
  auto end = end_tensor.item<int64_t>();
  StageSubmission::getInstance().setStageSubmissionFlow();

  // Handle case for all False where we return empty tensor with size
  if (end == 0) {
    auto shape = DimVector{0, dimensions};
    auto output =
        empty_hpu_lazy(shape, hb_options, self.suggest_memory_format(), true);
    auto hl_output = GetHbLazyTensor(output);
    updateDstDependencies(hl_output, output);
    flush_op(output);
    return output;
  }

  // Add a slice node to capture relevent elements from nonzero node
  // in case we have relevant elements
  auto result = slice_hpu_lazy(where_tensor, 0, 0, end, 1);
  flush_op(result);
  return result;
}

Tensor& nonzero_out_hpu_lazy(const Tensor& self, Tensor& output) {
  PT_LAZY_TRACE;
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  int elements = self.numel();
  at::TensorOptions hb_options = self.options();
  hb_options = hb_options.dtype(c10::ScalarType::Long);

  // Handle case for empty tensor where we return empty tensor with size
  if (elements == 0) {
    auto out_shape = DimVector{0, dimensions};
    auto hl_result = GetOrCreateHbLazyTensor(output, c10::kHPU);
    auto out_reshaped = hl_result.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    output.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));
    updateDstDependencies(hl_result, output);
    flush_op(output);
    return output;
  }

  // Add nonzero node
  std::vector<int64_t> output_shape{elements, dimensions};
  std::vector<int64_t> shape_tensor_shape{5};
  using T = std::tuple<at::Tensor, at::Tensor>;
  LazyOp<T> k(
      "hpu::nonzero",
      {self, c10::nullopt},
      {},
      {output_shape, shape_tensor_shape},
      0);
  // nonzero returns 2 output where and shape tensor
  auto result_nonzero = k.call();
  auto where_tensor = std::get<0>(result_nonzero);
  auto shape_tensor = std::get<1>(result_nonzero);

  // Select second element from shape tensor
  auto node_slice = std::make_shared<ir::Slice>(shape_tensor, 0, 1);
  auto end_shape = DimVector{1};
  auto end_tensor = empty_hpu_lazy(
      end_shape, hb_options, self.suggest_memory_format(), false);
  auto hl_end = GetHbLazyTensor(end_tensor);
  ir::Value& end_out = hl_end.CurrentIrValue();
  end_out.SetNode(
      node_slice,
      hl_end.GetDevice(),
      hl_end.GetSizes(),
      hl_end.dtype_optional());
  // Force an exections here to capture second element of shape tensor.
  // This element is required to determine shape of next node's output
  updateDstDependencies(hl_end, end_tensor);
  std::vector<HbLazyTensor> hl_flush_end = {
      hl_end, GetHbLazyTensor(where_tensor), GetHbLazyTensor(shape_tensor)};
  PT_IRGRAPH_DEBUG("step marker due to non zero");
  HbLazyTensor::SyncTensorsGraph(&hl_flush_end);
  auto cpu_end_tensor = end_tensor.to(c10::kCPU);
  auto end = cpu_end_tensor.item<int64_t>();
  StageSubmission::getInstance().setStageSubmissionFlow();

  // Handle case for all False where we return empty tensor with size
  if (end == 0) {
    auto sliced_shape = DimVector{0, dimensions};
    auto hl_result = GetOrCreateHbLazyTensor(output, c10::kHPU);
    auto out_reshaped = hl_result.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, sliced_shape.size(), sliced_shape.data(), nullptr);
    output.unsafeGetTensorImpl()->set_sizes_contiguous(
        IntArrayRef(sliced_shape));
    updateDstDependencies(hl_result, output);
    flush_op(output);
    return output;
  }

  // Add a slice node to capture relevent elements from nonzero node
  // in case we have relevant elements
  auto out_shape = DimVector{end, dimensions};

  auto hl_result = GetOrCreateHbLazyTensor(output, c10::kHPU);
  auto out_reshaped = hl_result.getAttachedTensorImpl();
  THHTensor_resizeNd(out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  output.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));

  auto node = std::make_shared<ir::Slice>(where_tensor, 0, 0, end, 1);

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, output);
  flush_op(output);
  return output;
}

Tensor masked_select_hpu_lazy(const Tensor& self, const Tensor& mask) {
  PT_LAZY_TRACE;
  Tensor reshape_mask = mask;
  if (mask.dim() == 0) {
    reshape_mask = mask.unsqueeze(0);
  }
  // Broadcast mask tensor if necessary
  if (self.sizes().vec() != mask.sizes().vec()) {
    auto broadcast_shape = at::infer_size(self.sizes(), mask.sizes());
    reshape_mask = mask.broadcast_to(broadcast_shape);
  }
  auto result = nonzero_hpu_lazy(reshape_mask);

  std::vector<Tensor> idx = result.unbind(1);
  // after unbind indices might be on cpu.
  // Before passing it to index operator all indices must be on hpu
  // This is done as an alternative of typeConvertIndices
  c10::List<c10::optional<Tensor>> converted_inds;
  converted_inds.reserve(idx.size());
  for (size_t i = 0; i < idx.size(); ++i) {
    const auto& ind = idx[i];
    if (ind.defined()) {
      converted_inds.push_back(ind);
    } else {
      converted_inds.push_back(std::move(idx[i]));
    }
  }
  return index(self, converted_inds);
}

Tensor& masked_select_out_hpu_lazy(
    const Tensor& self,
    const Tensor& mask,
    Tensor& out) {
  PT_LAZY_TRACE;
  Tensor reshape_mask = mask;
  if (mask.dim() == 0) {
    reshape_mask = mask.unsqueeze(0);
  }
  // Broadcast mask tensor if necessary
  if (self.sizes().vec() != mask.sizes().vec()) {
    auto broadcast_shape = at::infer_size(self.sizes(), mask.sizes());
    reshape_mask = mask.broadcast_to(broadcast_shape);
  }
  auto result = nonzero_hpu_lazy(reshape_mask);

  std::vector<Tensor> idx = result.unbind(1);
  // after unbind indices might be on cpu.
  // Before passing it to index operator all indices must be on hpu
  // This is done as an alternative of typeConvertIndices
  c10::List<c10::optional<Tensor>> converted_inds;
  converted_inds.reserve(idx.size());
  for (size_t i = 0; i < idx.size(); ++i) {
    const auto& ind = idx[i];
    if (ind.defined()) {
      converted_inds.push_back(ind);
    } else {
      converted_inds.push_back(std::move(idx[i]));
    }
  }
  // Resize output tensor(s) to correct shape
  // Output shape is the 1st dim value of index result from non_zero
  auto hl_out = GetOrCreateHbLazyTensor(out, c10::kHPU);
  std::vector<int64_t> out_shape{result.sizes().vec()[0]};
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = hl_out.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));
  }
  auto output = index(self, converted_inds);
  out.copy_(output);
  return out;
}

Tensor& index_add_hpu_lazy_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& indices,
    const Tensor& source,
    const Scalar& alpha,
    Tensor& out) {
  PT_LAZY_TRACE;

  auto dim_ = at::maybe_wrap_dim(dim, self.dim(), true);

  LazyOp<Tensor> index_add_op(
      "aten::index_add",
      {self, dim_, indices, source, alpha},
      {1}, // metadata_indices
      {self.sizes().vec()} // out_shapes
  );

  Tensor index_add_out = index_add_op.call();

  LazyOp<at::Tensor&> k{"hpu::habana_d2d_memcpy_other", {index_add_out, out}};
  return k.call(out);
}

Tensor& index_add_hpu_lazy_(
    Tensor& self,
    int64_t dim,
    const Tensor& indices,
    const Tensor& source) {
  PT_LAZY_TRACE;

  // TPC doesn't support inplace index add natively
  // Implement using out of place index add followed by D2D copy
  // TODO revisit once strided mem copy feature is mature
  auto dim_ = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  auto hl_self = GetOrCreateHbLazyTensor(self);

  LazyOp<Tensor> index_add_op(
      "aten::index_add",
      {self, dim_, indices, source},
      {1}, // metadata_indices
      {self.sizes().vec()} // out_shapes
  );

  Tensor index_add_out = index_add_op.call();

  LazyOp<at::Tensor&> k{"hpu::habana_d2d_memcpy_other", {index_add_out, self}};
  return k.call(self);
}

Tensor index_put_frontend_impl_hpu_lazy(
    const Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices_list,
    const Tensor& value_in,
    bool accumulate) {
  PT_LAZY_TRACE;
  std::vector<at::Tensor> indices_vec;
  for (c10::optional<Tensor> input : indices_list) {
    indices_vec.push_back(input.value());
  }
  std::vector<Tensor> indices_vec_out{};
  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].device().type() != c10::DeviceType::HPU) {
      indices_vec[i] = indices_vec[i].to(c10::kHPU);
    }
  }
  // handle views for tensorlist indices
  TensorList indices_in_list(indices_vec);
  indices_vec = HbLazyTensorViews::HandleViewsTensorList(indices_in_list);
  // for case where indices are Boolean tensor(s), convert these to integer
  // indices using nonzero operator before calling index
  if (indices_vec[0].scalar_type() == c10::ScalarType::Bool) {
    // do a mark_step to avoid attaching the select + scatter to a larger
    // previous graph
    HbLazyTensor::StepMarker({});
    for (size_t i = 0; i < indices_vec.size(); i++) {
      auto list = torch::nonzero_numpy(indices_vec.at(i));
      habana_lazy::SyncAccThreadPool();
      indices_vec_out.insert(
          indices_vec_out.cend(), list.cbegin(), list.cend());
    }
  }
  at::TensorList indices =
      (indices_vec[0].scalar_type() == c10::ScalarType::Bool) ? indices_vec_out
                                                              : indices_vec;
  auto indices_out_vec = HbLazyTensorViews::HandleViewsTensorList(indices);
  TensorList indices_out_list(indices_out_vec);
  // Assuming if 1st indices tensor is ZST then other indices tensors in list
  // (if any) will be ZST too. For ZST indices tensor broadcast and scatter_nd
  // operations are throwing GC errors therefore we have this workaround to
  // return a copy of input tensor.
  // TBD: Investigate further and raise a JIRA on GC.
  if (indices_out_list[0].numel() == 0 || value_in.numel() == 0) {
    auto result = self.clone();
    auto hl_result = GetHbLazyTensor(result);
    updateDstDependencies(hl_result, result);
    flush_op(result);
    return result;
  }

  // Broadcast indices
  auto broadcasted_indices = at::broadcast_tensors(indices_out_list);
  auto shape_broadcasted = broadcasted_indices[0].sizes().vec();
  // Reshape broadcasted indices to [N, 1] for concatenation
  auto flattened_size = std::accumulate(
      std::begin(shape_broadcasted),
      std::end(shape_broadcasted),
      1,
      std::multiplies<size_t>());
  std::vector<at::Tensor> flattened_idx;
  for (auto b : broadcasted_indices)
    flattened_idx.push_back(at::reshape(b, {flattened_size, 1}));
  // Create index tensor of shape [num_updates, dimensionality of indices]
  auto concatenated_indices = at::cat(flattened_idx, -1);
  // additional casts inserted for handling dtypes other than f32/bf16 because
  // scatter_nd TPC kernels used supports only f32/bf16
  at::Tensor self_cast = self;
  at::Tensor value = value_in;

  if (self.scalar_type() != c10::ScalarType::Double &&
      self.scalar_type() != c10::ScalarType::Float &&
      self.scalar_type() != c10::ScalarType::BFloat16 &&
      self.scalar_type() != c10::ScalarType::Long &&
      self.scalar_type() != c10::ScalarType::Int) {
    // i8/i16/i32 -> f32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {self, c10::ScalarType::Int},
        {self.sizes().vec()},
        c10::ScalarType::Int};
    self_cast = k_.call();
    // i8/i16/i32 -> f32
    LazyOp<at::Tensor> kv_{
        "hpu::cast",
        {value_in, c10::ScalarType::Int},
        {value_in.sizes().vec()},
        c10::ScalarType::Int};
    value = kv_.call();
  }

  // Calculate the dimensionality of updates for broadcasting
  auto rank_inp = self.ndimension();
  auto rank_idx = concatenated_indices.sizes().vec()[1];
  std::vector<int64_t> value_upd_dim{concatenated_indices.sizes().vec()[0]};
  for (int i = rank_idx; i < rank_inp; i++)
    value_upd_dim.push_back(self.sizes().vec()[i]);
  auto broadcasted_values = value.broadcast_to(value_upd_dim);
  if (!accumulate) {
    LazyOp<Tensor> scatter_nd_op(
        "hpu::scatter_nd_onnx",
        {self_cast, concatenated_indices, broadcasted_values});
    Tensor scatter_nd_out = scatter_nd_op.call();
    if (self.scalar_type() != c10::ScalarType::Double &&
        self.scalar_type() != c10::ScalarType::Float &&
        self.scalar_type() != c10::ScalarType::BFloat16 &&
        self.scalar_type() != c10::ScalarType::Long &&
        self.scalar_type() != c10::ScalarType::Int) {
      auto out_type = (self.scalar_type() == c10::ScalarType::Long)
          ? (c10::ScalarType::Int)
          : self.scalar_type();
      LazyOp<at::Tensor> k_{
          "hpu::cast",
          {scatter_nd_out, out_type},
          {scatter_nd_out.sizes().vec()},
          out_type};
      return k_.call();
    }
    return scatter_nd_out;
  } else {
    // Convert indices to values (ravelling indices) for sorting
    std::vector<int64_t> indices_shape;
    for (int i = 0; i < concatenated_indices.sizes().vec()[1]; i++)
      indices_shape.push_back(self_cast.sizes().vec()[i]);
    // Compute multiplication factor for each dimension
    std::vector<int> mul_factor_v{1};
    for (size_t i = 0; i < indices_shape.size() - 1; i++) {
      mul_factor_v.push_back(mul_factor_v[i] * indices_shape[i]);
    }
    auto mul_factor =
        torch::from_blob(
            mul_factor_v.data(), {1, int64_t(mul_factor_v.size())}, torch::kInt)
            .to(c10::kHPU, true);
    auto multiplied_indices = at::mul(concatenated_indices, mul_factor);
    auto ravelled_indices = at::sum(multiplied_indices, 1);
    auto sorted_results = at::sort(ravelled_indices, -1, true);
    auto permutation = std::get<1>(sorted_results).to(torch::kInt);
    auto grouped_indices =
        at::index_select(concatenated_indices, 0, permutation);
    auto update_locs =
        at::reshape(permutation, {permutation.sizes().vec()[0], 1});
    LazyOp<Tensor> scatter_nd_onnx_op(
        "hpu::scatter_nd",
        {self_cast,
         concatenated_indices,
         grouped_indices,
         update_locs,
         broadcasted_values});
    Tensor scatter_nd_onnx_out = scatter_nd_onnx_op.call();
    auto result = at::add(self_cast, scatter_nd_onnx_out);

    if (self.scalar_type() != c10::ScalarType::Double &&
        self.scalar_type() != c10::ScalarType::Float &&
        self.scalar_type() != c10::ScalarType::BFloat16 &&
        self.scalar_type() != c10::ScalarType::Long &&
        self.scalar_type() != c10::ScalarType::Int) {
      auto out_type = (self.scalar_type() == c10::ScalarType::Long)
          ? (c10::ScalarType::Int)
          : self.scalar_type();
      LazyOp<at::Tensor> k_{
          "hpu::cast", {result, out_type}, {result.sizes().vec()}, out_type};
      return k_.call();
    }
    return result;
  }
}

std::vector<Tensor> nonzero_ip_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  int elements = self.numel();
  at::TensorOptions hb_options = self.options();
  hb_options = hb_options.dtype(c10::ScalarType::Int);

  // Handle case for empty tensor where we return empty tensor with size
  if (elements == 0) {
    auto shape = DimVector{0, dimensions};
    auto output =
        empty_hpu_lazy(shape, hb_options, self.suggest_memory_format(), true);
    auto hl_output = GetHbLazyTensor(output);
    updateDstDependencies(hl_output, output);
    flush_op(output);
    return {output, output};
  }

  using T = std::tuple<at::Tensor, at::Tensor>;
  struct NonZero : LazyOp<T> {
    explicit NonZero(
        const std::vector<at::IValue>& inputs,
        const std::set<size_t>& metadata_indices = {},
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor>>(
              "hpu::nonzero",
              inputs,
              metadata_indices,
              out_shapes,
              -1) {}

    std::tuple<at::Tensor, at::Tensor> get_result_overrideable() override {
      auto inputs = get_inputs();
      auto outputs = get_out_shapes();
      auto self = inputs[0].toTensor();
      auto where_tensor = empty_hpu_lazy(
          outputs[0],
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format(),
          true);
      auto shape_tensor = empty_hpu_lazy(
          outputs[1],
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format(),
          true);
      flush_op(where_tensor);
      flush_op(shape_tensor);
      return {where_tensor, shape_tensor};
    }
  };

  // Add nonzero node
  auto output_shape = NonZeroOperator::compute_output_shape(self);
  std::vector<int64_t> shape_tensor_shape{5};
  Tensor nz_shape_tensor;
  c10::optional<at::Tensor> nonzero_shape_tensor =
      c10::make_optional(nz_shape_tensor);
  NonZero k({self, c10::nullopt}, {}, {output_shape, shape_tensor_shape});
  // nonzero returns 2 output where and shape tensor
  auto result_nonzero = k.call();
  auto where_tensor = std::get<0>(result_nonzero);
  auto shape_tensor = std::get<1>(result_nonzero);
  return {where_tensor, shape_tensor};
}

Tensor index_put_hpu_lazy(
    const Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices_list,
    const Tensor& value_in,
    bool accumulate) {
  PT_LAZY_TRACE;
  std::vector<at::Tensor> indices_vec;
  for (c10::optional<Tensor> input : indices_list) {
    indices_vec.push_back(input.value());
  }
  TensorList indices_in(indices_vec);
  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].device().type() != c10::DeviceType::HPU) {
      indices_vec[i] = indices_vec[i].to(c10::kHPU);
    }
  }
  // handle views for tensorlist indices
  TensorList indices_in_list(indices_vec);
  indices_vec = HbLazyTensorViews::HandleViewsTensorList(indices_in_list);
  at::TensorList indices = indices_vec;
  // For ZST indices tensor scatter_nd
  // operation is throwing GC error therefore we have this workaround to
  // return a copy of input tensor.
  // GC Jira - SW-73941
  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].numel() == 0 || value_in.numel() == 0) {
      auto result = self.clone();
      auto hl_result = GetHbLazyTensor(result);
      updateDstDependencies(hl_result, result);
      flush_op(result);
      return result;
    }
  }
  if (indices_vec[0].scalar_type() == c10::ScalarType::Bool) {
    TensorList indices_in_list(indices_vec);
    indices_vec = HbLazyTensorViews::HandleViewsTensorList(indices_in_list);
    at::TensorList indices = indices_vec;
    auto nonzero_outputs = nonzero_ip_hpu_lazy(indices[0]);
    // We need to slice the output of nonzero
    // since the new CGUID flow returns a padded output.
    // In order to provide the right input tensor to
    // index_put, it must be with respect to the original
    // input tensor and not the padded output from non-zero
    auto nonzero_sliced_outputs =
        slice_hpu_lazy(nonzero_outputs[0], 0, 0, indices[0].numel(), 1);
    // Calculate the dimensionality of updates for broadcasting
    auto rank_inp = self.ndimension();
    auto rank_idx = nonzero_sliced_outputs.sizes().vec()[1];
    std::vector<int64_t> value_upd_dim;

    if ((value_in.numel() >
         1)) { // if values has more than 1 elem, we have to assume the valid
               // count in indices will match values numel
      if (indices[0].dim() != self.dim() &&
          value_in.dim() != (1 + (self.dim() - indices[0].dim()))) {
        value_upd_dim.push_back(nonzero_sliced_outputs.sizes().vec()[0]);
        for (int i = rank_idx; i < rank_inp; i++)
          value_upd_dim.push_back(self.sizes().vec()[i]);
      } else {
        for (int i = 0; i < value_in.dim(); i++)
          value_upd_dim.push_back(value_in.sizes().vec()[i]);
      }
    } else { // We are assuming uses passes value shapes correctly for scatter
      value_upd_dim.push_back(nonzero_sliced_outputs.sizes().vec()[0]);
      for (int i = rank_idx; i < rank_inp; i++)
        value_upd_dim.push_back(self.sizes().vec()[i]);
    }

    auto value_dim_tensor = empty_hpu_lazy(
        value_upd_dim,
        self.options(),
        self.suggest_memory_format(),
        false,
        SHAPE_TENSOR);
    auto zero_shape_tensor = empty_hpu_lazy(
        self.sizes(),
        self.options(),
        self.suggest_memory_format(),
        false,
        SHAPE_TENSOR);
    LazyOp<at::Tensor> index_put_op{
        "hpu::index_put",
        {self,
         nonzero_sliced_outputs,
         nonzero_outputs[1],
         value_in,
         value_dim_tensor,
         zero_shape_tensor,
         accumulate}};
    return index_put_op.call();
  }
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) ||
      GET_ENV_FLAG_NEW(PT_HPU_FORCE_INDEX_PUT_FRONTEND_FALLBACK)) {
    return index_put_frontend_impl_hpu_lazy(
        self, indices_list, value_in, accumulate);
  }

  LazyOp<at::Tensor> index_put_op{
      "aten::index_put", {self, indices, value_in, accumulate}};
  return index_put_op.call();
}

Tensor& index_put_hpu_lazy_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    bool accumulate) {
  PT_LAZY_TRACE;
  std::vector<at::Tensor> indices_in;
  for (c10::optional<Tensor> input : indices) {
    indices_in.push_back(input.value());
  }

  auto isIndicesBool = indices_in[0].scalar_type() == c10::ScalarType::Bool;
  auto self_clone = self;
  auto index_put_result =
      index_put_hpu_lazy(self_clone, indices, value, accumulate);

  LazyOp<at::Tensor&> k{
      "hpu::habana_d2d_memcpy_other", {index_put_result, self}};
  self = k.call(self);

  HbLazyTensorViews::HandleViewsD2D(index_put_result, self);
  // In DS case changing shapes will not cause a cache miss, therefore no need
  // to break index_put op from subsequent graph whereas in other cases changing
  // shapes will cause cache misses therefore breaking graph.
  if (GET_ENV_FLAG_NEW(PT_HPU_FORCE_INDEX_PUT_FRONTEND_FALLBACK) ||
      (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) &&
       isIndicesBool && (accumulate || value.dim()))) {
    std::vector<HbLazyTensor> hl_flush_end = {GetHbLazyTensor(self)};
    HbLazyTensor::SyncTensorsGraph(&hl_flush_end);
  } else {
    flush_op(self);
  }
  return self;
}

Tensor& index_fill_hpu_lazy_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value) {
  auto dim_ = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  if (dim_ == 0) {
    auto value_dim = self.sizes().vec();
    value_dim[0] = index.numel();
    auto value_tensor = empty_hpu_lazy(
        value_dim, self.options(), self.suggest_memory_format(), true);
    fill_hpu_lazy_(value_tensor, value);
    c10::List<c10::optional<at::Tensor>> indices;
    indices.push_back(index);
    return index_put_hpu_lazy_(self, indices, value_tensor, false);
  } else {
    std::vector<int64_t> permute_dims(self.dim());
    std::iota(permute_dims.begin(), permute_dims.end(), 0);
    auto temp = permute_dims[self.dim() - dim_ - 1];
    permute_dims[self.dim() - dim_ - 1] = permute_dims[self.dim() - 1];
    permute_dims[self.dim() - 1] = temp;
    auto permuted_self = permute_hpu_lazy(self, permute_dims);

    auto value_dim = permuted_self.sizes().vec();
    value_dim[0] = index.numel();
    auto value_tensor = empty_hpu_lazy(
        value_dim, self.options(), self.suggest_memory_format(), true);
    fill_hpu_lazy_(value_tensor, value);

    c10::List<c10::optional<at::Tensor>> indices;
    indices.push_back(index);
    permuted_self =
        index_put_hpu_lazy_(permuted_self, indices, value_tensor, false);
    permuted_self = permute_hpu_lazy(permuted_self, permute_dims);
    LazyOp<at::Tensor&> k{
        "hpu::habana_d2d_memcpy_other", {permuted_self, self}};
    return k.call(self);
  }
}

Tensor& index_copy_hpu_lazy_(
    Tensor& self,
    int64_t dim,
    const Tensor& indices,
    const Tensor& source) {
  PT_LAZY_TRACE;
  // TPC doesn't support inplace index add natively
  // Implement using out of place index add followed by D2D copy
  // TODO revisit once strided mem copy feature is mature
  auto dim_ = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  auto hl_self = GetOrCreateHbLazyTensor(self);

  LazyOp<Tensor> index_copy_op(
      "aten::index_copy",
      {self, dim_, indices, source},
      {1}, // metadata_indices
      {self.sizes().vec()} // out_shapes
  );

  Tensor index_copy_out = index_copy_op.call();

  LazyOp<at::Tensor&> k{"hpu::habana_d2d_memcpy_other", {index_copy_out, self}};
  return k.call(self);
}

Tensor& masked_scatter_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& source) {
  PT_LAZY_TRACE;
  auto broadcasted_mask = mask.broadcast_to(self.sizes().vec());
  auto flattened_size = std::accumulate(
      std::begin(source.sizes()),
      std::end(source.sizes()),
      1,
      std::multiplies<size_t>());
  auto flattened_values = at::reshape(source, {flattened_size});
  c10::List<c10::optional<at::Tensor>> indices;
  indices.push_back(broadcasted_mask);
  return index_put_hpu_lazy_(self, indices, flattened_values, false);
}

Tensor slice_hpu_lazy(
    const Tensor& self_in,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  PT_LAZY_TRACE;
  habana_lazy::SyncAccThreadPool();
  auto hl_self_in = GetHbLazyTensor(self_in);

  // Native fork implementation to slice op is introduced to
  // allocate correct autograd gradient function for view tensor.
  auto out = at::native::slice(self_in, dim, start, end, step);
  auto hb_result = GetHbLazyTensor(out);
  {
    auto context = habana_lazy_executor.getDeviceExecutionContext(0);
    LOCK_VIEW_TABLE_MUTEX(context->viewContext);
    auto strided_param = HbLazyTensorViews::getViewTableParams(hb_result);
    // There could be some cases where view/select/etc followed by slice, in
    // those cases use as_strided instead of using the SliceOP.
    if (is_fallback_original_op(self_in, out)) {
      strided_param->optype = kStridedOpSlice;
      StridedOpSliceParams slice_param = {dim, start, end, step};
      strided_param->params.slice_param = slice_param;

      PT_VIEWTABLE_DEBUG(
          "slice fallback tensor id ",
          hl_self_in.getTensorUniqueId(),
          " dim ",
          dim,
          " start ",
          start.has_value() ? start.value() : 0,
          " end ",
          end.has_value() ? end.value() : -1,
          " step ",
          step);
    }
  }
  return out;
}

Tensor alias_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  auto hl_self_in = GetHbLazyTensor(self);
  auto out = as_strided_hpu_lazy(
      self, self.sizes(), self.strides(), self.storage_offset());
  auto hb_result = GetHbLazyTensor(out);
  {
    auto context = habana_lazy_executor.getDeviceExecutionContext(0);
    LOCK_VIEW_TABLE_MUTEX(context->viewContext);
    auto strided_param = HbLazyTensorViews::getViewTableParams(hb_result);
    if (is_fallback_original_op(self, out)) {
      strided_param->optype = kStridedOpIdentity;

      PT_VIEWTABLE_DEBUG(
          "alias-identity fallback tensor id ", hl_self_in.getTensorUniqueId());
    }
  }
  return out;
}

Tensor select_hpu_lazy(const Tensor& self, int64_t dim, int64_t index) {
  PT_LAZY_TRACE;
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "select() cannot be applied to a 0-dim tensor.");
  }
  dim = c10::maybe_wrap_dim(dim, ndim);
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    TORCH_CHECK_INDEX(
        false,
        "select(): index ",
        index,
        " out of range for tensor of size ",
        self.sizes(),
        " at dimension ",
        dim);
  }
  if (index < 0) {
    index += size;
  }

  c10::optional<int64_t> start_opt = c10::make_optional(index);

  int64_t end = index + 1;
  c10::optional<int64_t> end_opt = c10::make_optional(end);
  auto slice_out = slice_hpu_lazy(self, dim, start_opt, end_opt, 1);
  auto out = squeeze_hpu_lazy(slice_out, dim);

  // single op tests expect 0-D to be preserved at the front end.
  if (self.dim() == 1) {
    SET_SIZE_STRIDE_0D(out);
  }
  return out;
}

Tensor select_backward_hpu_lazy(
    const Tensor& grad,
    at::IntArrayRef input_sizes,
    int64_t dim,
    int64_t index) {
  PT_LAZY_TRACE;

  return at::native::select_backward(grad, input_sizes, dim, index);
}

bool can_convert(const Scalar& value) {
  if (value.isFloatingPoint()) {
    auto float_value = value.toFloat();
    auto int_value = value.toInt();
    auto diff = float_value - int_value;
    return !(diff > 0);
  }
  return true;
}

Tensor& arange_hpu_lazy_ht(
    Tensor& output,
    const Scalar& start,
    const Scalar& end,
    const Scalar& step) {
  PT_LAZY_TRACE;
  auto hl_result = GetOrCreateHbLazyTensor(output, c10::kHPU);

  // resizing the output as it is coming as empty from model
  int out_depth = ArangeOperator::GetOutputSize(start, end, step);
  auto out_shape = DimVector({out_depth});
  auto out_reshaped = hl_result.getAttachedTensorImpl();
  THHTensor_resizeNd(out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  output.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));

  ir::NodePtr node;
  std::vector<at::Tensor> input_pt_vec;

  // Currently synapse support dynamic shape arange only for int datatypes.
  // For any other output datatype, will fallback to normal flow.
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) &&
      ((start.isIntegral(false) || can_convert(start)) &&
       (end.isIntegral(false) || can_convert(end)) &&
       (step.isIntegral(false) || can_convert(step)))) {
    std::vector<int32_t> params_vec{start.toInt(), end.toInt(), step.toInt()};
    auto params_shape = empty_hpu_lazy(
        params_vec.size(),
        output.options(),
        output.suggest_memory_format(),
        false,
        HOST_TO_DEVICE_TENSOR);
    auto hl_params_shape = GetOrCreateHbLazyTensor(params_shape, c10::kHPU);

    auto hl_param_internal = hl_params_shape.CurrentTensorAttached().value();
    habana_lazy::HbInternalTensorImpl* impl =
        habana_lazy::GetHbInternalTensorImpl(hl_param_internal);
    HABANA_ASSERT(impl);
    impl->set_host_data(
        params_vec.data(),
        params_vec.size(),
        sizeof(int),
        HostDataType::INT32_T);

    // Create a dummy shape tensor for the output, this shape tensor is not
    // added to synapse graph, but only ensures that when we match in bucket
    // we are restricted by the size of the output
    auto result_shape = empty_hpu_lazy(
        out_shape,
        output.options(),
        c10::MemoryFormat::Contiguous,
        false,
        SHAPE_TENSOR);
    // Mark this front end shape tensor as it does not need synapse tensor
    auto hl_result_shape = GetOrCreateHbLazyTensor(result_shape, c10::kHPU);
    auto hl_result_shape_internal =
        hl_result_shape.CurrentTensorAttached().value();
    auto stImpl =
        habana_lazy::GetHbInternalTensorImpl(hl_result_shape_internal);
    if (stImpl) {
      stImpl->setH2DFrontEndShapeTensor();
    }

    if (output.scalar_type() == c10::ScalarType::Int ||
        output.scalar_type() == c10::ScalarType::Long) {
      node = ir::Node::Create(
          Symbol::fromQualString("hpu::arange_out_ds_ht"),
          {hl_params_shape.GetIrValue(),
           hl_result.GetIrValue(),
           hl_result_shape.GetIrValue()});
      ir::Value& out = hl_result.CurrentIrValue();
      out.SetNode(
          node,
          hl_result.GetDevice(),
          hl_result.GetSizes(),
          hl_result.dtype_optional());
      input_pt_vec.emplace_back(output);
      input_pt_vec.emplace_back(params_shape);
      input_pt_vec.emplace_back(result_shape);
      node->AddInputPtTensors(input_pt_vec);
    } else {
      // If result is not int, capture the result in int and add cast node
      auto int_output = empty_hpu_lazy(
          IntArrayRef(out_shape),
          output.options().dtype(c10::ScalarType::Int),
          output.suggest_memory_format(),
          true);
      auto hl_int_output = GetOrCreateHbLazyTensor(int_output, c10::kHPU);
      node = ir::Node::Create(
          Symbol::fromQualString("hpu::arange_out_ds_ht"),
          {hl_params_shape.GetIrValue(),
           hl_int_output.GetIrValue(),
           hl_result_shape.GetIrValue()});
      ir::Value& out = hl_int_output.CurrentIrValue();
      out.SetNode(
          node,
          hl_int_output.GetDevice(),
          hl_int_output.GetSizes(),
          hl_int_output.dtype_optional());
      input_pt_vec.emplace_back(int_output);
      input_pt_vec.emplace_back(params_shape);
      input_pt_vec.emplace_back(result_shape);
      node->AddInputPtTensors(input_pt_vec);

      // Add cast node to cast int_output as required
      ir::Value& out_cast = hl_result.CurrentIrValue();
      ir::NodePtr node_cast =
          std::make_shared<ir::Cast>(int_output, output.scalar_type(), true);
      out_cast.SetNode(
          node_cast,
          hl_result.GetDevice(),
          hl_result.GetSizes(),
          hl_result.dtype_optional());
    }
    // updatet the view if any
    updateDstDependencies(hl_result, output);
    flush_op(output);
    return output;
  }
  auto hl_start = GetIrValueForScalar(start);
  auto hl_end = GetIrValueForScalar(end);
  auto hl_step = GetIrValueForScalar(step);
  node = ir::Node::Create(
      Symbol::fromQualString("hpu::arange_out"),
      {hl_start, hl_end, hl_step, hl_result.GetIrValue()});

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  input_pt_vec.emplace_back(output);
  node->AddInputPtTensors(input_pt_vec);
  // updatet the view if any
  updateDstDependencies(hl_result, output);
  flush_op(output);
  return output;
}

Tensor& arange_hpu_lazy(
    Tensor& output,
    const Scalar& start,
    const Scalar& end,
    const Scalar& step) {
  PT_LAZY_TRACE;

  if (GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR)) {
    return arange_hpu_lazy_ht(output, start, end, step);
  }

  auto hl_result = GetOrCreateHbLazyTensor(output, c10::kHPU);

  // resizing the output as it is coming as empty from model
  int out_depth = ArangeOperator::GetOutputSize(start, end, step);
  auto out_shape = DimVector({out_depth});
  auto out_reshaped = hl_result.getAttachedTensorImpl();
  THHTensor_resizeNd(out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  output.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));

  ir::NodePtr node;
  std::vector<at::Tensor> input_pt_vec;

  // Currently synapse support dynamic shape arange only for int datatypes.
  // For any other output datatype, will fallback to normal flow.
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) &&
      ((start.isIntegral(false) || can_convert(start)) &&
       (end.isIntegral(false) || can_convert(end)) &&
       (step.isIntegral(false) || can_convert(step)))) {
    std::vector<int64_t> params_vec{step.toInt(), end.toInt(), start.toInt()};
    auto input_size = IntArrayRef(params_vec.data(), params_vec.size());
    auto params_shape = empty_hpu_lazy(
        input_size,
        output.options(),
        output.suggest_memory_format(),
        false,
        INPUT_DESCRIBING_SHAPE_TENSOR);
    auto hl_params_shape = GetOrCreateHbLazyTensor(params_shape, c10::kHPU);
    if (output.scalar_type() == c10::ScalarType::Int ||
        output.scalar_type() == c10::ScalarType::Long) {
      node = ir::Node::Create(
          Symbol::fromQualString("hpu::arange_out_ds"),
          {hl_params_shape.GetIrValue(), hl_result.GetIrValue()});
      ir::Value& out = hl_result.CurrentIrValue();
      out.SetNode(
          node,
          hl_result.GetDevice(),
          hl_result.GetSizes(),
          hl_result.dtype_optional());
      input_pt_vec.emplace_back(output);
      input_pt_vec.emplace_back(params_shape);
      node->AddInputPtTensors(input_pt_vec);
    } else {
      // If result is not int, capture the result in int and add cast node
      auto int_output = empty_hpu_lazy(
          IntArrayRef(out_shape),
          output.options().dtype(c10::ScalarType::Int),
          output.suggest_memory_format(),
          true);
      auto hl_int_output = GetOrCreateHbLazyTensor(int_output, c10::kHPU);
      node = ir::Node::Create(
          Symbol::fromQualString("hpu::arange_out_ds"),
          {hl_params_shape.GetIrValue(), hl_int_output.GetIrValue()});
      ir::Value& out = hl_int_output.CurrentIrValue();
      out.SetNode(
          node,
          hl_int_output.GetDevice(),
          hl_int_output.GetSizes(),
          hl_int_output.dtype_optional());
      input_pt_vec.emplace_back(int_output);
      input_pt_vec.emplace_back(params_shape);
      node->AddInputPtTensors(input_pt_vec);

      // Add cast node to cast int_output as required
      ir::Value& out_cast = hl_result.CurrentIrValue();
      ir::NodePtr node_cast =
          std::make_shared<ir::Cast>(int_output, output.scalar_type(), true);
      out_cast.SetNode(
          node_cast,
          hl_result.GetDevice(),
          hl_result.GetSizes(),
          hl_result.dtype_optional());
    }
    // updatet the view if any
    updateDstDependencies(hl_result, output);
    flush_op(output);
    return output;
  }
  auto hl_start = GetIrValueForScalar(start);
  auto hl_end = GetIrValueForScalar(end);
  auto hl_step = GetIrValueForScalar(step);
  node = ir::Node::Create(
      Symbol::fromQualString("hpu::arange_out"),
      {hl_start, hl_end, hl_step, hl_result.GetIrValue()});

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  input_pt_vec.emplace_back(output);
  node->AddInputPtTensors(input_pt_vec);
  // updatet the view if any
  updateDstDependencies(hl_result, output);
  flush_op(output);
  return output;
}

Tensor kl_div_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k(
      "aten::kl_div",
      {self, target, reduction, log_target},
      {2, 3}, // metadata_indices
      {KlDivOperator::compute_output_shape(self, reduction)});
  RUN_MAYBE_WITH_ACC_THREAD(kl_div, k)
}

Tensor kl_div_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k(
      "aten::kl_div_backward",
      {grad_output, self, target, reduction, log_target},
      {3, 4}, // metadata_indices
      {self.sizes().vec()});
  RUN_MAYBE_WITH_ACC_THREAD(kl_div_backward, k)
}

/*
For (N,C,L) inputs, reshape to (N,C,1,L) in PyTorch framework order
For (N,C) inputs, reshape to (N,C,1,1) in Pytorch Framework order
*/
static inline Tensor bn_reshape_to_4d(const Tensor& in_t) {
  Tensor reshaped_t;
  std::vector<int64_t> ret_shape(4, 1);
  auto in_shape = in_t.sizes().vec();
  Tensor in_ = in_t;
  // TPC  BN supports only 4D inputs. This means that any higher dims have to be
  // flattened
  if (in_shape.size() > 4) {
    // Input is in dims format N,C,D1,D2,D3,...,Dm,H,W
    std::vector<int64_t> permute_dims(in_shape.size(), 0);
    for (size_t i = 0; i < permute_dims.size(); i++) {
      permute_dims[i] = i;
    }
    std::swap(permute_dims[1], permute_dims[permute_dims.size() - 3]);
    // Changed/Permuted Input is in dims format N,Dm,D1,D2,D3,...,C,H,W
    in_ = in_t.permute(permute_dims);
    in_shape = in_.sizes().vec();
    auto higher_dim_size = std::accumulate(
        in_shape.begin(),
        in_shape.begin() + in_shape.size() - 3,
        1,
        std::multiplies<int64_t>{});
    ret_shape[0] = higher_dim_size;
    // Get the shape ready to change input to format
    // {(N*Dm*D1*D2*D3*Dm-1),C,H,W}
    std::copy(
        in_shape.begin() + in_shape.size() - 3,
        in_shape.end(),
        ret_shape.begin() + 1);
  } else {
    std::copy(in_shape.begin(), in_shape.end(), ret_shape.begin());
  }
  if (3 == in_shape.size()) { // For 3-D in_t[2] should be at reshaped_t[3]
    std::swap(ret_shape[2], ret_shape[3]);
  }
  reshaped_t = in_.reshape(ret_shape);
  return reshaped_t;
}

static inline Tensor bn_reshape_from_4d_to_orig(
    const Tensor& in_t,
    std::vector<int64_t> in_sizes) {
  Tensor res;
  int dims = in_sizes.size();
  switch (dims) {
    case 1:
      res = in_t.reshape({in_sizes[0]});
      break;
    case 2:
      res = in_t.reshape({in_sizes[0], in_sizes[1]});
      break;
    case 3:
      res = in_t.reshape({in_sizes[0], in_sizes[1], in_sizes[2]});
      break;
    default:
      // Input is in dims format N,Dm,D1,D2,D3,...,C,H,W
      // Final output should be in format N,C,D1,D2,D3,...,Dm,H,W
      std::vector<int64_t> permute_dims(in_sizes.size(), 0);
      for (size_t i = 0; i < permute_dims.size(); i++) {
        permute_dims[i] = i;
      }
      std::swap(permute_dims[1], permute_dims[permute_dims.size() - 3]);
      std::swap(in_sizes[1], in_sizes[in_sizes.size() - 3]);
      auto res_ = in_t.reshape(in_sizes);
      res = res_.permute(permute_dims);
      break;
  }
  return res;
}

static inline Tensor bn_create_and_init_undefined_input(
    const Tensor& in_t,
    c10::MemoryFormat memfmt,
    bool fill,
    Scalar val) {
  IntArrayRef rm_size;
  Tensor ret_t;
  if (memfmt == c10::MemoryFormat::ChannelsLast) {
    rm_size = in_t.sizes()[3];
  } else if (memfmt == c10::MemoryFormat::ChannelsLast3d) {
    rm_size = in_t.sizes()[4];
  } else {
    rm_size = in_t.sizes()[1];
  }
  // undefined inputs are model params which are always in Float32
  ret_t = empty_hpu_lazy(
      rm_size,
      in_t.options().dtype(c10::ScalarType::Float),
      in_t.suggest_memory_format(),
      true);
  if (fill)
    fill_hpu_lazy_(ret_t, val);
  return ret_t;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm_fwd_preprocess(
    const Tensor& input_,
    const Tensor& weight_tensor,
    const Tensor& bias_tensor,
    const Tensor& running_mean_,
    const Tensor& running_var_) {
  Tensor input;
  auto in_sizes = input_.sizes().vec();
  if (input_.ndimension() != 4) {
    input = bn_reshape_to_4d(input_);
  } else {
    input = input_;
  }

  Tensor running_mean, running_var;
  // if RMV are undefined, create zero mean and unit variance tensors for
  // numerical stability of BN. Note that they should have same
  // dtype as weight

  auto weight = weight_tensor;
  auto bias = bias_tensor;
  if (!weight.defined()) {
    weight = bn_create_and_init_undefined_input(
        input, input.suggest_memory_format(), true, 1);
  }

  if (!bias.defined()) {
    bias = bn_create_and_init_undefined_input(
        input, input.suggest_memory_format(), true, 0);
  }

  if (!running_mean_.defined()) {
    running_mean = bn_create_and_init_undefined_input(
        input, input.suggest_memory_format(), true, 0);
  } else {
    running_mean = running_mean_;
  }

  if (!running_var_.defined()) {
    running_var = bn_create_and_init_undefined_input(
        input, input.suggest_memory_format(), true, 1);
  } else {
    running_var = running_var_;
  }

  return {input, weight, bias, running_mean, running_var};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _batch_norm_fwd_training(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  PT_LAZY_TRACE;
  using T = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;
  struct BN : LazyOp<T> {
    BN(const Stack& inputs)
        : LazyOp<T>("hpu::native_batch_norm_training", inputs, {}, -1) {}
    T get_result_overrideable() override {
      const auto& inputs = get_inputs();
      const auto& input = inputs[0].toTensor();
      const auto& running_mean = inputs[3].toTensor();
      const auto& running_var = inputs[4].toTensor();
      auto result_img = empty_hpu_lazy(
          input.sizes(), input.options(), input.suggest_memory_format(), false);
      auto result_mean = empty_hpu_lazy(
          running_mean.sizes(),
          running_mean.options(),
          input.suggest_memory_format(),
          false);
      auto result_var = empty_hpu_lazy(
          running_var.sizes(),
          running_var.options(),
          input.suggest_memory_format(),
          false);
      return {result_img, result_mean, result_var, running_mean, running_var};
    }
  };
  BN op(
      {input,
       bias,
       weight,
       running_mean,
       running_var,
       training,
       momentum,
       eps});
  return op.call();
}

Tensor _batch_norm_fwd_inference(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  PT_LAZY_TRACE;
  LazyOp<Tensor> op(
      "hpu::native_batch_norm_inf",
      {input,
       bias,
       weight,
       running_mean,
       running_var,
       training,
       momentum,
       eps});
  return op.call();
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_hpu_lazy(
    const Tensor& input_,
    const c10::optional<at::Tensor>& weight_tensor,
    const c10::optional<at::Tensor>& bias_tensor,
    const c10::optional<at::Tensor>& running_mean_,
    const c10::optional<at::Tensor>& running_var_,
    bool training,
    double momentum,
    double eps) {
  PT_LAZY_TRACE;
  auto in_sizes = input_.sizes().vec();
  auto running_tensor_mean = running_mean_.value_or(Tensor());
  auto preprocess_results = batch_norm_fwd_preprocess(
      input_,
      weight_tensor.value_or(Tensor()),
      bias_tensor.value_or(Tensor()),
      running_tensor_mean,
      running_var_.value_or(Tensor()));

  auto input = std::get<0>(preprocess_results);
  auto weight = std::get<1>(preprocess_results);
  auto bias = std::get<2>(preprocess_results);
  auto running_mean = std::get<3>(preprocess_results);
  auto running_var = std::get<4>(preprocess_results);

  bool inference_mode = !training && running_tensor_mean.defined();
  if (!inference_mode) { /*training mode*/
    auto res_ = _batch_norm_fwd_training(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        !inference_mode, // we don't rely just on the training flag from
                         // PyTorch
        momentum,
        eps);
    Tensor res;
    auto res0 = std::get<BNFwdTPCRetIndex::Output>(res_);
    if (input_.ndimension() != 4) {
      res = bn_reshape_from_4d_to_orig(res0, in_sizes);
    } else {
      res = res0;
    }
    return {
        res,
        std::get<BNFwdTPCRetIndex::SavedMean>(res_),
        std::get<BNFwdTPCRetIndex::SavedIStd>(res_)};
  } else {
    auto res_ = _batch_norm_fwd_inference(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        !inference_mode,
        momentum,
        eps);
    Tensor res;
    if (input_.ndimension() != 4) {
      res = bn_reshape_from_4d_to_orig(res_, in_sizes);
    } else {
      res = res_;
    }
    return {res, running_mean, running_var};
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm_bwd_preprocess(
    const Tensor& input_,
    const Tensor& grad_out_,
    const Tensor& weight_tensor,
    const Tensor& running_mean_,
    const Tensor& running_var_) {
  Tensor input;
  Tensor grad_out;
  auto in_sizes = input_.sizes().vec();
  auto gradout_sizes = grad_out_.sizes().vec();
  if (input_.ndimension() != 4) {
    input = bn_reshape_to_4d(input_);
  } else {
    input = input_;
  }
  if (grad_out_.ndimension() != 4) {
    grad_out = bn_reshape_to_4d(grad_out_);
  } else {
    grad_out = grad_out_;
  }
  auto weight = weight_tensor;
  if (!weight.defined()) {
    weight = bn_create_and_init_undefined_input(
        input, input.suggest_memory_format(), true, 1);
  }

  Tensor running_mean, running_var;
  // create tensors if RMV are undefined. Note that they should have same
  // dtype as weight
  if (!running_mean_.defined()) {
    running_mean = bn_create_and_init_undefined_input(
        input, input.suggest_memory_format(), false, 0);
  } else {
    running_mean = running_mean_;
  }

  if (!running_var_.defined()) {
    running_var = bn_create_and_init_undefined_input(
        input, input.suggest_memory_format(), false, 1);
  } else {
    running_var = running_var_;
  }
  return {input, grad_out, weight, running_mean, running_var};
}

std::tuple<Tensor, Tensor, Tensor> _batch_norm_bwd(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    bool not_train_rm) {
  PT_LAZY_TRACE;
  Tensor mean = save_mean;
  Tensor invstd = save_invstd;
  if (not_train_rm) {
    mean = running_mean;
    invstd = at::rsqrt(at::add(running_var, eps));
  }
  using T = std::tuple<Tensor, Tensor, Tensor>;
  struct BN : LazyOp<T> {
    BN(const Stack& inputs)
        : LazyOp<T>("hpu::native_batch_norm_backward", inputs, {}, -1) {}
    T get_result_overrideable() override {
      const auto& inputs = get_inputs();
      auto input = inputs[0].toTensor();
      auto mean = inputs[2].toTensor();
      auto invstd = inputs[3].toTensor();
      auto create_res = [&](Tensor in) {
        Tensor res;
        // first output is based on input and for HPU-TPC implementation it
        // cannot be left uncreated.
        // We ignore output_mask values as TPC always creates 3 outputs
        res = empty_hpu_lazy(
            in.sizes(), in.options(), in.suggest_memory_format(), false);
        return res;
      };
      return {create_res(input), create_res(mean), create_res(invstd)};
    }
  };
  BN op({input, grad_out, mean, invstd, weight, train, eps, 0.0 /*momentum*/});
  return op.call();
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd_hpu_lazy(
    const Tensor& grad_out_,
    const Tensor& input_,
    const c10::optional<at::Tensor>& weight_tensor,
    const c10::optional<at::Tensor>& running_mean_,
    const c10::optional<at::Tensor>& running_var_,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd,
    bool train,
    double eps,
    UNUSED std::array<bool, 3> output_mask) {
  PT_LAZY_TRACE;
  auto in_sizes = input_.sizes().vec();
  auto running_tensor_mean = running_mean_.value_or(Tensor());
  auto preprocess_results = batch_norm_bwd_preprocess(
      input_,
      grad_out_,
      weight_tensor.value_or(Tensor()),
      running_tensor_mean,
      running_var_.value_or(Tensor()));

  auto input = std::get<0>(preprocess_results);
  auto grad_out = std::get<1>(preprocess_results);
  auto weight = std::get<2>(preprocess_results);
  auto running_mean = std::get<3>(preprocess_results);
  auto running_var = std::get<4>(preprocess_results);

  auto not_train_rm = !train && running_tensor_mean.defined();
  auto res_ = _batch_norm_bwd(
      grad_out,
      input,
      weight,
      running_mean,
      running_var,
      save_mean.value_or(Tensor()),
      save_invstd.value_or(Tensor()),
      train,
      eps,
      not_train_rm);
  auto res0 = std::get<0>(res_);
  Tensor res;
  if (input_.ndimension() != 4) {
    res = bn_reshape_from_4d_to_orig(res0, in_sizes);
  } else {
    res = res0;
  }
  return {
      res,
      std::get<BNBwdTPCRetIndex::WeightGrad>(res_) /*gamma*/,
      std::get<BNBwdTPCRetIndex::BiasGrad>(res_) /*beta*/
  };
}

::std::tuple<Tensor, Tensor> batch_norm_stats_lazy(
    const Tensor& input,
    double eps) {
  std::vector<int64_t> dim = {0, 2, 3};
  if (input.dim() == 5)
    dim.push_back(4);
  auto mean = at::mean(input, dim);
  auto var = at::var(input, dim, true);
  auto inv_std = at::reciprocal(at::sqrt(at::add(var, eps)));
  return std::tie(mean, inv_std);
}

Tensor batch_norm_elemt_lazy(
    const Tensor& input,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps) {
  static_cast<void>(eps);
  auto C = input.sizes().vec()[1];
  std::vector<int64_t> dim = {1, C, 1, 1};
  if (input.dim() == 5)
    dim.push_back(1);
  Tensor gamma, beta;
  if (weight.has_value())
    gamma = weight.value();
  else
    gamma = at::ones(C).to(torch::kHPU);

  if (bias.has_value())
    beta = bias.value();
  else
    beta = at::zeros(C).to(torch::kHPU);

  auto mean_reshaped = at::reshape(mean, dim);
  auto inv_std_reshaped = at::reshape(invstd, dim);
  auto gamma_reshaped = at::reshape(gamma, dim);
  auto beta_reshaped = at::reshape(beta, dim);

  auto out = at::add(
      at::mul(
          at::mul(at::sub(input, mean_reshaped), inv_std_reshaped),
          gamma_reshaped),
      beta_reshaped);
  return out;
}

Tensor batch_norm_backward_elemt_lazy(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& weight,
    const Tensor& mean_dy,
    const Tensor& mean_dy_xmu,
    const Tensor& count) {
  std::vector<int64_t> dim = {1, mean.sizes().vec()[0], 1, 1};
  if (input.dim() == 5)
    dim.push_back(1);

  auto mean_reshaped = at::reshape(mean, dim);
  auto invstd_reshaped = at::reshape(invstd, dim);
  auto mean_dy_reshaped = at::reshape(mean_dy, dim);
  auto mean_dy_xmu_reshaped = at::reshape(mean_dy_xmu, dim);
  auto total_count = at::sum(count);
  Tensor factor_2_c, factor_1_c;
  if (weight.has_value()) {
    factor_2_c = at::mul(weight.value(), invstd);
  } else {
    factor_2_c = at::reciprocal(invstd_reshaped);
  }

  factor_1_c = at::div(
      at::mul(at::mul(mean_dy_xmu_reshaped, invstd_reshaped), invstd_reshaped),
      total_count);

  auto grad_in = at::mul(
      at::sub(
          at::sub(grad_out, at::div(mean_dy_reshaped, total_count)),
          at::mul(at::sub(input, mean_reshaped), factor_1_c)),
      factor_2_c);
  return grad_in;
}

::std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_lazy(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& weight,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  static_cast<void>(weight);
  auto grad_out_reshaped = at::reshape(
      grad_out, {grad_out.sizes().vec()[0], grad_out.sizes().vec()[1], -1});
  auto mean_reshaped = at::reshape(mean, {1, mean.sizes().vec()[0], 1});
  auto inp_reshaped =
      at::reshape(input, {input.sizes().vec()[0], input.sizes().vec()[1], -1});
  auto invstd_reshaped = at::reshape(invstd, {1, invstd.sizes().vec()[0], 1});
  std::vector<int64_t> dim = {0, 2};
  Tensor sum_dy, sum_dy_xmu, grad_wei, grad_bias;
  sum_dy = input_g ? at::sum(grad_out_reshaped, dim) : sum_dy;

  auto dy_xmu =
      at::mul(grad_out_reshaped, at::sub(inp_reshaped, mean_reshaped));
  sum_dy_xmu = input_g ? at::sum(dy_xmu, dim) : sum_dy_xmu;
  auto wei_term = at::mul(dy_xmu, invstd_reshaped);
  grad_wei = weight_g ? at::sum(wei_term, dim) : grad_wei;
  grad_bias = bias_g ? at::sum(grad_out_reshaped, dim) : grad_bias;
  return std::tie(sum_dy, sum_dy_xmu, grad_wei, grad_bias);
}

::std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_lazy(
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
    double momentum,
    double eps,
    const Tensor& counts) {
  auto counts_reshaped = at::reshape(counts, {-1, 1});

  auto counts_accum_inclusive = at::cumsum(counts_reshaped, 0);

  auto counts_accum_exclusive =
      at::sub(counts_accum_inclusive, counts_reshaped);

  auto mean_times_counts = at::mul(counts_reshaped, mean);

  auto one_div_counts_accum_inclusive = at::reciprocal(counts_accum_inclusive);

  auto partial_mean =
      at::mul(at::cumsum(mean_times_counts, 0), one_div_counts_accum_inclusive);

  auto tmp_partial_mean = at::roll(partial_mean, 1, 0);
  auto type = kLong;
  auto const_tensor = empty_hpu_lazy(
      {1}, input.options().dtype(type), input.suggest_memory_format(), true);
  auto value_tensor = empty_hpu_lazy(
      {1},
      tmp_partial_mean.options(),
      tmp_partial_mean.suggest_memory_format(),
      true);
  fill_hpu_lazy_(const_tensor, 0);
  fill_hpu_lazy_(value_tensor, 0);
  index_put_hpu_lazy_(tmp_partial_mean, {const_tensor}, value_tensor, 0);

  auto second_term = at::mul(
      at::mul(
          at::mul(
              at::sub(tmp_partial_mean, mean), at::sub(tmp_partial_mean, mean)),
          at::mul(counts_accum_exclusive, counts_reshaped)),
      one_div_counts_accum_inclusive);

  auto v = at::reciprocal(invstd);
  auto w = at::mul(at::sub(at::mul(v, v), eps), counts_reshaped);

  auto first_term = at::cumsum(w, 0);
  auto partial_var = at::add(first_term, second_term);
  auto const_tensor2 = empty_hpu_lazy(
      {1}, input.options().dtype(type), input.suggest_memory_format(), true);
  fill_hpu_lazy_(const_tensor2, partial_var.sizes().vec()[0] - 1);
  auto partial_var_value = at::index_select(partial_var, 0, const_tensor2);
  auto partial_mean_value = at::index_select(partial_mean, 0, const_tensor2);
  auto counts_accum_value =
      at::index_select(counts_accum_inclusive, 0, const_tensor2);

  auto partial_var_reshaped =
      at::reshape(partial_var_value, {partial_var_value.numel()});
  auto g_invstd = at::reciprocal(
      at::sqrt(at::add(at::div(partial_var_value, counts_accum_value), eps)));

  auto out1 = at::reshape(partial_mean_value, {partial_mean_value.numel()});
  auto out2 = at::reshape(g_invstd, {g_invstd.numel()});

  if (running_mean.has_value()) {
    auto x = at::mul(out1, momentum);
    running_mean.value().mul_(1 - momentum);
    running_mean.value().add_(x);
  }
  if (running_var.has_value()) {
    auto unbiasedVar =
        at::div(partial_var_reshaped, at::sub(at::sum(counts), 1));
    auto x = at::mul(unbiasedVar, momentum);
    running_var.value().mul_(1 - momentum);
    running_var.value().add_(x);
  }
  return std::tie(out1, out2);
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_hpu_lazy(
    const Tensor& input,
    IntArrayRef normalized_shape_,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    double eps) {
  PT_LAZY_TRACE;
  auto weight = weight_opt.value_or(Tensor());
  auto sizes_vec = input.sizes().vec();
  // check whether we can use a perf optimized TPC exec path
  auto use_tpc_affine_path =
      LayerNormOperator::is_tpc_affine_path(input, normalized_shape_, weight);
  std::vector<int64_t> normalized_shape_vec = normalized_shape_.vec();
  if (use_tpc_affine_path) { // if optimized path, then we can't have
                             // N/mini-batch-size for generating weights/biases
    sizes_vec.erase(sizes_vec.begin());
    // NOTE: Add Hack to indicate to lowering kernel that
    // elementwise_affine=False Without this we have to change the schema and
    // add a new variable to indicate the path. If, in future, TPC moves fully
    // to use optimized path, we can remove this.
    if (normalized_shape_vec.size() == (size_t)input.dim() - 1) {
      normalized_shape_vec.insert(normalized_shape_vec.begin(), 1);
    }
  }
  IntArrayRef normalized_shape = normalized_shape_vec;
  if (!weight.defined()) {
    auto options = torch::TensorOptions()
                       .dtype(input.dtype())
                       .device(torch::kHPU)
                       .requires_grad(false);
    weight = torch::ones(normalized_shape_vec, options);
  }

  auto bias = bias_opt.value_or(Tensor());
  if (!bias.defined()) {
    auto options = torch::TensorOptions()
                       .dtype(input.dtype())
                       .device(torch::kHPU)
                       .requires_grad(false);
    bias = torch::zeros(normalized_shape_vec, options);
  }

  ir::NodePtr node = std::make_shared<ir::LayerNormForward>(
      input, normalized_shape, weight, bias, eps);

  auto sizes = LayerNormOperator::getOutputSizes(input, normalized_shape);
  LazyOp<std::tuple<Tensor, Tensor, Tensor>, ir::LayerNormForward> k{
      node, {input, normalized_shape, weight, bias, eps}, sizes};
  return k.call();
}
std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_hpu_lazy(
    const at::Tensor& dY,
    const at::Tensor& X,
    IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    std::array<bool, 3> grad_input_mask) {
  PT_LAZY_TRACE;
  ir::NodePtr node = std::make_shared<ir::LayerNormBackward>(
      dY,
      X,
      normalized_shape,
      mean,
      rstd,
      weight_opt,
      bias_opt,
      grad_input_mask);
  // Get Output Image
  using T = std::tuple<Tensor, Tensor, Tensor>;
  using U = ir::LayerNormBackward;
  class Kernel : public LazyOp<T, U> {
   public:
    Kernel(
        ir::NodePtr node,
        const at::Tensor& dY,
        const at::Tensor& X,
        IntArrayRef normalized_shape,
        const at::Tensor& mean,
        const at::Tensor& rstd,
        const c10::optional<Tensor>& weight_opt,
        const c10::optional<Tensor>& bias_opt,
        std::array<bool, 3> grad_input_mask)
        : LazyOp<T, U>(
              std::move(node),
              {dY,
               X,
               mean,
               rstd,
               weight_opt,
               bias_opt,
               normalized_shape,
               grad_input_mask},
              {6, 7},
              {},
              -1),
          dY{dY},
          normalized_shape{normalized_shape},
          weight_opt{weight_opt},
          grad_input_mask{grad_input_mask} {}

   private:
    T get_result_overrideable() override {
      auto gamma = weight_opt.value_or(Tensor());
      auto sizes = LayerNormBackwardOperator::getOutputSizes(dY, gamma);
      auto result_dY = empty_hpu_lazy(
          sizes[0], dY.options(), dY.suggest_memory_format(), false);
      at::Tensor result2, result3;
      if (grad_input_mask[1]) {
        result2 = empty_hpu_lazy(
            sizes[1], gamma.options(), gamma.suggest_memory_format(), false);
      }
      if (grad_input_mask[2]) {
        result3 = empty_hpu_lazy(
            sizes[2], gamma.options(), gamma.suggest_memory_format(), false);
      }
      return std::make_tuple(result_dY, result2, result3);
    }
    const at::Tensor& dY;
    IntArrayRef normalized_shape;
    const c10::optional<Tensor>& weight_opt;
    std::array<bool, 3> grad_input_mask;
  };

  Kernel k(
      node,
      dY,
      X,
      normalized_shape,
      mean,
      rstd,
      weight_opt,
      bias_opt,
      grad_input_mask);
  return k.call();
}
Tensor fill_0d_val(const Tensor& self, const c10::Scalar& val) {
  std::vector<int64_t> size = {};
  at::Tensor empty_tensor =
      empty_hpu_lazy(size, self.options(), self.suggest_memory_format(), true);
  return fill_hpu_lazy_(empty_tensor, val);
}

std::tuple<Tensor, Tensor, Tensor> instance_norm_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    double eps) {
  PT_LAZY_TRACE;

  auto mean_var_shape = InstanceNormOperator::compute_output_shape(
      input, c10::MemoryFormat::Contiguous);

  using T = std::tuple<Tensor, Tensor, Tensor>;
  LazyOp<T> k(
      "hpu::instance_norm",
      {input, weight, bias, eps},
      {3}, // metadata_indices
      {input.sizes().vec(), mean_var_shape, mean_var_shape} // out_shapes
  );

  T results = k.call();
  return results;
}

std::tuple<Tensor, Tensor, Tensor> instance_norm_backward_hpu_lazy(
    const Tensor& input,
    const Tensor& grad_in,
    const Tensor& mean,
    const Tensor& istd,
    const Tensor& gamma) {
  PT_LAZY_TRACE;

  auto grad_beta_gamma_shape =
      InstanceNormBackwardOperator::compute_output_shape(
          input, c10::MemoryFormat::Contiguous);
  using T = std::tuple<Tensor, Tensor, Tensor>;
  LazyOp<T> k(
      "hpu::instance_norm_backward",
      {input, grad_in, mean, istd, gamma},
      {}, // metadata_indices
      {input.sizes().vec(), grad_beta_gamma_shape, grad_beta_gamma_shape}
      // out_shapes
  );

  T results = k.call();
  return results;
}

std::tuple<Tensor, Tensor> max_pool2d_with_indices_hpu_lazy(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  PT_LAZY_TRACE;
  ir::NodePtr maxpool_node = std::make_shared<ir::MaxPool>(
      input, kernel_size, stride, padding, dilation, ceil_mode);
  using T = std::tuple<Tensor, Tensor>;
  using U = ir::MaxPool;
  class Kernel : public LazyOp<T, U> {
   public:
    Kernel(
        ir::NodePtr node,
        const Tensor& input,
        IntArrayRef kernel_size,
        IntArrayRef stride,
        IntArrayRef padding,
        IntArrayRef dilation,
        bool ceil_mode)
        : LazyOp<T, U>(
              std::move(node),
              {input, kernel_size, stride, padding, dilation, ceil_mode},
              {1, 2, 3, 4, 5},
              {},
              -1),
          input{std::move(input)},
          kernel_size{std::move(kernel_size)},
          stride{std::move(stride)},
          padding{std::move(padding)},
          dilation{std::move(dilation)},
          ceil_mode{std::move(ceil_mode)} {}

   private:
    T get_result_overrideable() override {
      // shape inferrence
      auto opsize_nhwc = PoolHelper::compute_output_shape(
          input, kernel_size, stride, padding, dilation, ceil_mode, false);

      std::vector<long int> shape_out = opsize_nhwc;
      if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
        // return always nhwc. convert to nchw
        shape_out = {
            opsize_nhwc.at(0),
            opsize_nhwc.at(3),
            opsize_nhwc.at(1),
            opsize_nhwc.at(2)};
      }
      // allocate Output_0 storage
      auto result_0 = empty_hpu_lazy(
          shape_out, input.options(), input.suggest_memory_format(), false);

      // allocate Output_1 storage
      auto type = kByte;
      if (input.scalar_type() == c10::ScalarType::BFloat16) {
        type = kShort;
      }
      auto result_1 = empty_hpu_lazy(
          shape_out,
          input.options().dtype(type),
          input.suggest_memory_format(),
          false);
      return {result_0, result_1};
    }
    const Tensor& input;
    IntArrayRef kernel_size;
    IntArrayRef stride;
    IntArrayRef padding;
    IntArrayRef dilation;
    bool ceil_mode;
  };

  Kernel k(
      maxpool_node, input, kernel_size, stride, padding, dilation, ceil_mode);
  return k.call();
}

Tensor& max_pool2d_with_indices_backward_out_hpu_lazy(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  PT_LAZY_TRACE;
  FALLBACK_IF_UNSUPPORTED_OP2_O(
      max_pool2d_with_indices_backward,
      PARAMS2(
          grad_output,
          input,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          indices,
          grad_input),
      grad_input)
}

Tensor max_pool2d_with_indices_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  PT_LAZY_TRACE;
  ir::NodePtr maxpool_bwd_node = std::make_shared<ir::MaxPoolBackWard>(
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices);

  // shape inferrence
  // since grad_input should match memory format only checking for input
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, false);
  std::vector<long int> out_shape = opsize_nhwc;
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    // retunr always nhwc. convert to nchw
    out_shape = {
        opsize_nhwc.at(0),
        opsize_nhwc.at(3),
        opsize_nhwc.at(1),
        opsize_nhwc.at(2)};
  }

  TORCH_CHECK(grad_output.sizes().vec() == out_shape);
  TORCH_CHECK(
      (indices.scalar_type() == c10::ScalarType::Byte) ||
      (indices.scalar_type() == c10::ScalarType::Short));

  LazyOp<at::Tensor, ir::MaxPoolBackWard> k{
      maxpool_bwd_node,
      {grad_output,
       input,
       kernel_size,
       stride,
       padding,
       dilation,
       ceil_mode,
       indices},
      {2, 3, 4, 5, 6},
      {input.sizes().vec()}};
  return k.call();
}

Tensor adaptive_avg_pool2d_hpu_lazy(
    const Tensor& input,
    IntArrayRef output_size) {
  PT_LAZY_TRACE;
  auto opsize_nhwc =
      PoolHelper::compute_output_shape(input, output_size, false);
  std::vector<long int> shape_out = opsize_nhwc;
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    // compute_output_shape return always nhwc. convert to nchw
    shape_out = {
        opsize_nhwc.at(0),
        opsize_nhwc.at(3),
        opsize_nhwc.at(1),
        opsize_nhwc.at(2)};
  }

  LazyOp<Tensor> k{
      "aten::_adaptive_avg_pool2d", {input, output_size}, {1}, {shape_out}};
  RUN_MAYBE_WITH_ACC_THREAD(adaptive_avg_pool2d, k)
}

Tensor adaptive_avg_pool2d_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<Tensor> k{
      "aten::_adaptive_avg_pool2d_backward",
      {grad_output, input},
      {},
      {input.sizes().vec()}};
  RUN_MAYBE_WITH_ACC_THREAD(adaptive_avg_pool2d_backward, k)
}

void randperm_hpu_lazy_ht(
    Tensor& output,
    int64_t n,
    c10::optional<Generator> gen) {
  PT_LAZY_TRACE;
  auto out_shape = DimVector({n});
  std::vector<int32_t> params_vec{0 /*start*/, (int32_t)n /*end*/, 1 /*step*/};
  auto params_shape = empty_hpu_lazy(
      params_vec.size(),
      output.options(),
      output.suggest_memory_format(),
      false,
      HOST_TO_DEVICE_TENSOR);
  auto hl_params_shape = GetOrCreateHbLazyTensor(params_shape, c10::kHPU);
  auto hl_param_internal = hl_params_shape.CurrentTensorAttached().value();
  habana_lazy::HbInternalTensorImpl* impl =
      habana_lazy::GetHbInternalTensorImpl(hl_param_internal);
  impl->set_host_data(
      params_vec.data(),
      params_vec.size(),
      sizeof(int32_t),
      HostDataType::INT32_T);
  auto result_shape = empty_hpu_lazy(
      out_shape,
      output.options(),
      c10::MemoryFormat::Contiguous,
      false,
      SHAPE_TENSOR);
  auto hl_result_shape = GetOrCreateHbLazyTensor(result_shape, c10::kHPU);

  LazyOp<Tensor&> op{
      "hpu::randperm_out_ds_ht",
      {params_shape, result_shape, std::move(gen), output},
      {2},
      {},
      3};
  op.call(output);
}

Tensor& randperm_hpu_lazy(
    int64_t n,
    c10::optional<Generator> gen,
    Tensor& output) {
  PT_LAZY_TRACE;
  auto func = [output, n, gen = std::move(gen)]() mutable {
    // resizing the output as it is coming as empty from model
    auto hl_result = GetOrCreateHbLazyTensor(output, c10::kHPU);
    auto out_shape = DimVector({n});
    auto out_reshaped = hl_result.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    output.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));

    // Currently synapse support dynamic shape arange only for int datatypes.
    // For any other output datatype, will fallback to normal flow.
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) &&
        (output.scalar_type() == c10::ScalarType::Int ||
         output.scalar_type() == c10::ScalarType::Long)) {
      if (GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_RANDPERM_HOST_TENSOR)) {
        randperm_hpu_lazy_ht(output, n, gen);
      } else {
        std::vector<int64_t> params_vec{1 /*step*/, n /*end*/, 0 /*start*/};
        auto input_size = IntArrayRef(params_vec.data(), params_vec.size());
        auto params_shape = empty_hpu_lazy(
            input_size,
            output.options(),
            output.suggest_memory_format(),
            false,
            INPUT_DESCRIBING_SHAPE_TENSOR);
        LazyOp<Tensor&> op{
            "hpu::randperm_out_ds",
            {params_shape, std::move(gen), output},
            {},
            {},
            2};
        op.call(output);
      }
    } else {
      LazyOp<Tensor&> op{
          "hpu::randperm_out",
          {Scalar((int32_t)n), std::move(gen), output},
          {1},
          {{n}}};
      op.call(output);
    }
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(randperm_out, func, output)
}

std::tuple<Tensor, Tensor> fused_dropout_hpu_lazy(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen) {
  PT_LAZY_TRACE;
  struct FusedDropout : LazyOp<std::tuple<Tensor, Tensor>> {
    FusedDropout(const Tensor& self, double p, const Tensor& seed)
        : LazyOp<std::tuple<Tensor, Tensor>>(
              "hpu::_fused_dropout",
              {self, p, seed},
              {},
              {},
              -1) {}

    std::tuple<Tensor, Tensor> get_result_overrideable() override {
      auto t = get_inputs().at(0).toTensor();
      at::Tensor result0 = empty_hpu_lazy(
          t.sizes(), t.options(), t.suggest_memory_format(), false);
      at::Tensor result1 = empty_hpu_lazy(
          t.sizes(),
          t.options().dtype(c10::ScalarType::Char),
          t.suggest_memory_format(),
          false);
      return {result0, result1};
    }
  };
  // use gen to create a seed and forward it to the op
  auto seed = habana::get_seed_tensor_hpu(gen);
  FusedDropout op(self, p, seed);
  return op.call();
}

at::Tensor repeat_hpu_lazy_ht(const at::Tensor& self, at::IntArrayRef repeats) {
  PT_LAZY_TRACE;
  std::vector<at::IValue> vector_of_inputs;
  std::string op_name;
  std::set<size_t> metadata_indices;

  auto rpt_vec = repeats.vec();
  std::vector<int32_t> params_vec;
  for_each(rpt_vec.rbegin(), rpt_vec.rend(), [&](const int64_t& n) {
    params_vec.push_back(static_cast<int32_t>(n));
  });
  auto params_shape = empty_hpu_lazy(
      params_vec.size(),
      self.options(),
      self.suggest_memory_format(),
      false,
      HOST_TO_DEVICE_TENSOR);

  auto hl_params_shape = GetOrCreateHbLazyTensor(params_shape, c10::kHPU);

  auto hl_param_internal = hl_params_shape.CurrentTensorAttached().value();

  habana_lazy::HbInternalTensorImpl* impl =
      habana_lazy::GetHbInternalTensorImpl(hl_param_internal);
  HABANA_ASSERT(impl);

  impl->set_host_data(
      params_vec.data(),
      params_vec.size(),
      sizeof(int32_t),
      HostDataType::INT32_T);
  auto out_shape = RepeatOperator::compute_output_shape(self, repeats);
  std::vector<int64_t> repeat_shape(rpt_vec.rbegin(), rpt_vec.rend());
  auto repeat_shape_tensor = empty_hpu_lazy(
      repeat_shape,
      self.options(),
      c10::MemoryFormat::Contiguous,
      false,
      SHAPE_TENSOR);

  // Mark this front end shape tensor as it does not need synapse tensor
  auto repeat_internal = GetOrCreateHbLazyTensor(repeat_shape_tensor, c10::kHPU)
                             .CurrentTensorAttached()
                             .value();
  auto stImpl = habana_lazy::GetHbInternalTensorImpl(repeat_internal);
  if (stImpl) {
    stImpl->setH2DFrontEndShapeTensor();
  }
  vector_of_inputs = {self, params_shape, repeat_shape_tensor};
  op_name = "hpu::repeat_ht";
  metadata_indices = {};
  LazyOp<at::Tensor> k{
      op_name, vector_of_inputs, metadata_indices, {out_shape}};
  return k.call();
}

at::Tensor repeat_hpu_lazy(const at::Tensor& self, at::IntArrayRef repeats) {
  PT_LAZY_TRACE;

  if (GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_REPEAT_HOST_TENSOR)) {
    return repeat_hpu_lazy_ht(self, repeats);
  }

  std::vector<at::IValue> vector_of_inputs;
  std::string op_name;
  std::set<size_t> metadata_indices;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    auto repeats_shape = empty_hpu_lazy(
        repeats,
        self.options(),
        self.suggest_memory_format(),
        false,
        INPUT_DESCRIBING_SHAPE_TENSOR);

    vector_of_inputs = {self, repeats_shape};
    op_name = "hpu::repeat";
    metadata_indices = {};
  } else {
    vector_of_inputs = {self, repeats};
    op_name = "aten::repeat";
    metadata_indices = {1};
  }

  LazyOp<at::Tensor> k{
      op_name,
      vector_of_inputs,
      metadata_indices,
      {RepeatOperator::compute_output_shape(self, repeats)}};
  RUN_MAYBE_WITH_ACC_THREAD(repeat, k)
}

at::Tensor repeat_inlv_hpu_lazy(
    const at::Tensor& repeats,
    c10::optional<int64_t> output_size) {
  // if output_size is not provided by user, there is no way to compute output
  // shape without peeking into the "repeats" tensor. See desc. from PyT docs,
  // "output_size (int, optional) – Total output size for the given axis ( e.g.
  // sum of repeats). If given, it will avoid stream syncronization needed to
  // calculate output shape of the tensor."

  // In our case because of the use of H2D tensor for repeats, we will always
  // break the graph if model puts repeats tensor on HPU, but this should be ok
  // as this will not cause a blocking synchronization
  int64_t out_size;
  // repeats can only by "long" or "int", if long, cast to int because synapse
  // cannot handle long tensors
  bool need_h2d_tensor = false;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) ||
      !output_size.has_value()) {
    need_h2d_tensor = true;
  }
  at::Tensor repeats_cpu;
  if (need_h2d_tensor) {
    repeats_cpu = repeats.to("cpu").to(torch::kInt32);
  }
  if (output_size.has_value()) {
    out_size = output_size.value();
  } else {
    auto out = repeats_cpu.sum();
    out_size = out.item().toInt();
  }
  auto input = at::native::arange(
      repeats.sizes()[0],
      c10::ScalarType::
          Int, // c10::optTypeMetaToScalarType(repeats.options().dtype_opt()),
      repeats.options().layout_opt(),
      repeats.options().device_opt(),
      repeats.options().pinned_memory_opt());

  at::Tensor repeats_tensor;
  if (need_h2d_tensor) {
    repeats_tensor = empty_hpu_lazy(
        repeats.sizes(),
        repeats.options().dtype(c10::ScalarType::Int),
        repeats.suggest_memory_format(),
        false,
        HOST_TO_DEVICE_TENSOR);
    auto hl_params_shape = GetOrCreateHbLazyTensor(repeats_tensor, c10::kHPU);

    auto hl_param_internal = hl_params_shape.CurrentTensorAttached().value();
    habana_lazy::HbInternalTensorImpl* impl =
        habana_lazy::GetHbInternalTensorImpl(hl_param_internal);
    HABANA_ASSERT(impl);
    impl->set_host_data(
        repeats_cpu.data_ptr(),
        repeats_cpu.sizes()[0],
        sizeof(int32_t),
        HostDataType::INT32_T);
  }

  auto output_shape =
      RepeatInlvOperator::compute_output_shape(input, 0, out_size);
  auto output_shape_tensor = empty_hpu_lazy(
      IntArrayRef(output_shape),
      input.options().dtype(c10::ScalarType::Int),
      input.suggest_memory_format(),
      false,
      SHAPE_TENSOR);
  // Mark this front end shape tensor as it does not need synapse tensor
  auto hl_output_shape_tensor =
      GetOrCreateHbLazyTensor(output_shape_tensor, c10::kHPU);
  auto hl_output_shape_tensor_internal =
      hl_output_shape_tensor.CurrentTensorAttached().value();
  auto stImpl =
      habana_lazy::GetHbInternalTensorImpl(hl_output_shape_tensor_internal);
  if (stImpl) {
    stImpl->setH2DFrontEndShapeTensor();
  }

  if (need_h2d_tensor) {
    LazyOp<at::Tensor> k{
        "hpu::repeat_inlv",
        {input, repeats_tensor, 0, output_shape_tensor},
        {2},
        {output_shape}};
    return k.call();
  } else {
    LazyOp<at::Tensor> k{
        "hpu::repeat_inlv",
        {input, repeats, 0, output_shape_tensor},
        {2},
        {output_shape}};
    return k.call();
  }
}

Tensor sum_dim_IntList_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;

  at::Tensor self_updated_dtype = self;
  ScalarType result_dtype;
  /* Match the HPU return dtype with CPU behaviour.
   * If 'dtype' parameter is specified and it has value:
   *      Return result in the dtype as specified by 'dtype' parameter.
   * Else:
   * Return result in:
   * i.  int64 for inputs of integral or bool dtypes
   * ii. corresponding floating point dtype for inputs
   *      of FP type.
   *      ie, fp32 input -> fp32 output
   *      ie, bf16 input -> bf16 output
   */
  if (dtype.has_value()) {
    if (dtype.value() != self_updated_dtype.scalar_type()) {
      self_updated_dtype = self.to(dtype.value());
    }
    result_dtype = dtype.value();
  } else {
    result_dtype = c10::isIntegralType(self_updated_dtype.scalar_type(), true)
        ? c10::ScalarType::Long
        : self_updated_dtype.scalar_type();
  }
  /* for non-floating types, tpc supports only int dtype for sum */
  if (c10::isIntegralType(self_updated_dtype.scalar_type(), true)) {
    self_updated_dtype = self_updated_dtype.to(c10::ScalarType::Int);
  }
  std::vector<at::IValue> vector_of_inputs;
  vector_of_inputs = {self_updated_dtype, dim, keepdim, dtype};

  using T = at::Tensor;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(
        const std::vector<at::IValue>& vector_of_inputs,
        ScalarType result_dtype)
        : LazyOp<T>("hpu::sum_dim_IntList", vector_of_inputs, {}, {}, -1),
          result_dtype_(result_dtype) {}

   private:
    T get_result_overrideable() override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      auto dim = inputs[1].toIntList();
      auto keepdim = inputs[2].toBool();
      auto shape =
          ReduceOperator::compute_output_shape(self, dim.vec(), keepdim);
      return empty_hpu_lazy(
          shape,
          self.options().dtype(result_dtype_),
          self.suggest_memory_format(),
          false);
    }
    ScalarType result_dtype_;
  };

  Kernel kernel{vector_of_inputs, result_dtype};
  return kernel.call();
}

Tensor mean_dim_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k(
      "aten::mean",
      {self, dim, keepdim, std::move(dtype)},
      {1, 2, 3}, // metadata_indices
      {ReduceOperator::compute_output_shape(self, dim, keepdim)});
  return k.call();
}

Tensor sum_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;

  at::Tensor self_updated_dtype = self;
  ScalarType result_dtype;

  /* Refer to comment in sum_dim_IntList_hpu_lazy on result dtype setting */
  if (dtype.has_value()) {
    if (dtype.value() != self_updated_dtype.scalar_type()) {
      self_updated_dtype = self.to(dtype.value());
    }
    result_dtype = dtype.value();
  } else {
    result_dtype = c10::isIntegralType(self_updated_dtype.scalar_type(), true)
        ? c10::ScalarType::Long
        : self_updated_dtype.scalar_type();
  }
  /* for non-floating types, tpc supports only int dtype for sum */
  if (c10::isIntegralType(self_updated_dtype.scalar_type(), true)) {
    self_updated_dtype = self_updated_dtype.to(c10::ScalarType::Int);
  }
  std::vector<at::IValue> vector_of_inputs;
  vector_of_inputs = {self_updated_dtype, dtype};
  using T = at::Tensor;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(
        const std::vector<at::IValue>& vector_of_inputs,
        ScalarType result_dtype)
        : LazyOp<T>("aten::sum", vector_of_inputs, {}, {}, -1),
          result_dtype_(result_dtype) {}

   private:
    T get_result_overrideable() override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      return empty_hpu_lazy(
          {},
          self.options().dtype(result_dtype_),
          self.suggest_memory_format(),
          false);
    }
    ScalarType result_dtype_;
  };

  Kernel kernel{vector_of_inputs, result_dtype};
  return kernel.call();
}

Tensor mean_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::mean", {self, dtype}, {}, {{}}};
  return k.call();
}

void InitSizesAndStrides(
    at::Tensor& at_tensor,
    c10::optional<synTensorType> tensor_type,
    c10::optional<IntArrayRef> size,
    c10::optional<IntArrayRef> stride,
    c10::optional<MemoryFormat> mem_format) {
  IntArrayRef tensor_size = size.value_or(at_tensor.sizes());

  if (stride.has_value()) {
    at_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        tensor_size, stride.value());
  } else if (
      tensor_type.has_value() && (tensor_type.value() == DEVICE_SHAPE_TENSOR)) {
    at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(
        device_shape_tensor_size);
  } else if ((4 == tensor_size.size()) && mem_format.has_value()) {
    at_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        tensor_size, CalculateStrides(tensor_size, mem_format.value()));
  } else if ((5 == tensor_size.size()) && mem_format.has_value()) {
    at_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        tensor_size, CalculateStrides5d(tensor_size, mem_format.value()));
  } else if (
      size.has_value() && (size.value().size() != 1 || size.value()[0] != 0)) {
    at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(tensor_size);
  }
}

Tensor empty_hpu_lazy(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format,
    bool create_storage,
    synTensorType tensor_type,
    c10::optional<std::reference_wrapper<const at::Tensor>> base_view) {
  PT_LAZY_TRACE;
  c10::optional<MemoryFormat> mem_format = optional_memory_format.has_value()
      ? optional_memory_format
      : options.memory_format_opt();
  auto original_dtype = options.dtype();
  auto type = typeMetaToScalarType(original_dtype);
  auto shape_tensor = habana_helpers::is_shape_tensor(tensor_type);
  TORCH_CHECK(
      options.pinned_memory() == false,
      "habana allocator doesn't supported pinned memory");
  c10::Allocator* allocator = habana::getHABANADeviceAllocator();
  HABANA_ASSERT(habana_helpers::is_supported_type(type));

  // Dont allocate 8 bytes for double/long as we are anyway going to cast at
  // CPU and then copy to device @ 4byts per element
  type = type == c10::ScalarType::Long ? c10::ScalarType::Int : type;
  type = type == c10::ScalarType::Double ? c10::ScalarType::Float : type;
  auto new_dtype = scalarTypeToTypeMeta(type);

  if (create_storage || shape_tensor) {
    int64_t nelements = multiply_integers(size);
    // we dont create a full storage for shape tensors but we need a backend
    // impl to get meta data
    if (shape_tensor) {
      nelements = (tensor_type == DEVICE_SHAPE_TENSOR) ? SYN_MAX_TENSOR_DIM : 0;
    }
    int elem_size = new_dtype.itemsize();
    int64_t size_bytes = nelements * elem_size;
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator->allocate(nelements * elem_size),
        allocator,
        /*resizeable=*/true);
    Tensor at_internal_tensor = AtenInternalHbTensor(
        std::move(storage_impl),
        new_dtype,
        tensor_type,
        size,
        c10::nullopt,
        mem_format);

    // backend tensor should always be contiguous as per view table design
    std::vector<int64_t> contig_strides = at_internal_tensor.strides().vec();
    if (contig_strides.size()) {
      habana_helpers::recalc_strides(
          contig_strides, at_internal_tensor.sizes().vec());
      IntArrayRef new_strides = contig_strides;
      at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          at_internal_tensor.sizes(), new_strides);
    }

    // set metadata that its a shape tensor
    if (shape_tensor) {
      habana_lazy::HbInternalTensorImpl* impl =
          habana_lazy::GetHbInternalTensorImpl(at_internal_tensor);
      if (impl) {
        impl->setTensorType(tensor_type);
      }
    }

    Tensor at_tensor;
    bool is_in_lowering_mode = false;
    if (habana_lazy_executor.getExecutionMode() ==
        LazyExecutionMode::kLOWERING) {
      is_in_lowering_mode = true;
    }

    // This call could have come from a .to call and not from a lowering
    // context. In such case, create the lazt tensor.
    if (!is_in_lowering_mode) {
      HbLazyTensor hb_tensor = HbLazyTensor::CreateHbLazyTensor(
          size, 0, options.device(), typeMetaToScalarType(original_dtype));
      at_tensor = AtenFromHbLazyTensor(
          hb_tensor, tensor_type, size, c10::nullopt, mem_format);

      // The lazy tensor will have a reference to the internal tensor
      hb_tensor.SetTensorData(at_internal_tensor);

      // Keep a pointer to the storageless tensor from the internal tensor
      auto at_internal_impl = GetHbInternalTensorImpl(at_internal_tensor);
      HABANA_ASSERT(at_internal_impl != nullptr);

      // Any lazy tensor created with storage should be marked as executed
      if (create_storage) {
        hb_tensor.getDataPtr()->execution_status = kEXECUTION_COMPLETE;
      }

      // As its an inplace op and we want this op to execute
      // we want to wind back status of this tensor to registered
      // so that when post order is created, we actually execute it
      // auto context =
      //    habana_lazy_executor.getDeviceExecutionContext(
      //        options.device().index());
      // context->MarkTensorStatus(
      //    hb_tensor.getDataPtr(),
      //    LazyTensorExecutionStatus::kINPUT);
      // setTensorAsInputNode(hb_tensor);
    }

    // If we are not from lowering context, return the storageless one.
    if (!is_in_lowering_mode) {
      return at_tensor;
    } else {
      // else return the internal tensor with storage
      return at_internal_tensor;
    }
  } else {
    HbLazyTensor hb_tensor = HbLazyTensor::CreateHbLazyTensor(
        size, 0, options.device(), typeMetaToScalarType(original_dtype));
    if (base_view.has_value()) {
      const auto& base = base_view.value().get();
      const auto& storage = base.storage();
      auto key_set = base.key_set();
      return (AtenFromHbLazyTensor(
          hb_tensor,
          storage,
          key_set,
          tensor_type,
          size,
          c10::nullopt,
          mem_format));
    } else {
      return (AtenFromHbLazyTensor(
          hb_tensor, tensor_type, size, c10::nullopt, mem_format));
    }
  }
}

Tensor empty_strided_hpu_lazy(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options,
    bool create_storage,
    synTensorType tensor_type,
    int64_t storage_offset,
    c10::optional<std::reference_wrapper<const at::Tensor>> base_view) {
  PT_LAZY_TRACE;
  at::Tensor empty_tensor = empty_hpu_lazy(
      size, options, c10::nullopt, create_storage, tensor_type, base_view);
  empty_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

  if (storage_offset) {
    empty_tensor.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  }

  // empty_hpu_lazy call might move the tensor to cpu for unsupported dtypes
  if (empty_tensor.device().type() != c10::DeviceType::HPU)
    return empty_tensor;
  // If we have created a tensor with storage, set the strides and sizes to
  // backend tensor as well
  if (create_storage) {
    auto hl_empty = TryGetHbLazyTensor(empty_tensor);
    if (hl_empty) {
      setTensorAsInputNode(hl_empty.value());
    }
  }
  return empty_tensor;
}

Tensor clone_hpu_lazy(
    const Tensor& self,
    c10::optional<MemoryFormat> /* memory_format */) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k{"hpu::habana_d2d_memcpy", {self}};
  auto result_func = [](at::Tensor& result) {
    result.unsafeGetTensorImpl()->set_sizes_contiguous(
        IntArrayRef(result.sizes()));
  };
  RUN_MAYBE_WITH_ACC_THREAD_MODIFY_RESULT(clone, k, result_func);
}

Tensor& zero_hpu_lazy(Tensor& self) {
  PT_LAZY_TRACE;
  return fill_hpu_lazy_(self, 0);
}

Tensor cat_hpu_lazy(const TensorList tensors, int64_t dim_) {
  PT_LAZY_TRACE;
  TORCH_CHECK(tensors.size() > 0, "Empty tensors list!");

  auto non_empty_list = filter(tensors, is_nonempty_tensor);
  auto first_tensor = tensors[0];

  if (non_empty_list.empty()) {
    return empty_hpu_lazy(
        first_tensor.sizes(),
        first_tensor.options(),
        first_tensor.suggest_memory_format(),
        true);
  }

  // calculate output shape
  auto output_shape =
      CatOutOperator::compute_output_shape(non_empty_list, dim_);

  // allocate output tensor
  auto out = empty_hpu_lazy(
      output_shape,
      first_tensor.options(),
      first_tensor.suggest_memory_format(),
      false);

  std::vector<Tensor> tensors_copy;
  std::copy(tensors.begin(), tensors.end(), std::back_inserter(tensors_copy));

  // parallel function that will be executed in the accumulation thread
  auto op_func = [tensors_ = std::move(tensors_copy),
                  dim_,
                  output_shape = std::move(output_shape),
                  out]() mutable {
    auto t_list = HbLazyTensorViews::HandleViewsTensorList(tensors_);
    t_list = filter(t_list, is_nonempty_tensor);
    const TensorList view_list{t_list};

    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
      auto output_shape_tensor = empty_hpu_lazy(
          IntArrayRef(output_shape),
          tensors_[0].options().dtype(c10::ScalarType::Int),
          tensors_[0].suggest_memory_format(),
          false,
          SHAPE_TENSOR);

      LazyOp<at::Tensor> k{
          "hpu::cat", {view_list, dim_, output_shape_tensor}, {1}, {}, 0};
      k.call(out);
    } else { // if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES))
      LazyOp<at::Tensor> k{"aten::cat", {view_list, dim_}, {1}, {}, 0};
      k.call(out);
    }
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(cat, op_func, out);
}

Tensor& cat_hpu_lazy_out(
    Tensor& result,
    const TensorList tensors,
    int64_t dim_) {
  if (filter(tensors, is_nonempty_tensor).empty()) {
    return result;
  }

  std::vector<Tensor> tensors_copy;
  std::copy(tensors.begin(), tensors.end(), std::back_inserter(tensors_copy));

  auto func = [result, tensors = std::move(tensors_copy), dim_]() mutable {
    auto t_list = HbLazyTensorViews::HandleViewsTensorList(tensors);
    t_list = filter(t_list, is_nonempty_tensor);
    if (t_list.empty())
      return;

    const TensorList view_list{t_list};

    auto out_size = CatOutOperator::compute_output_shape(t_list, dim_);
    LazyOp<at::Tensor&> k{"aten::cat", {view_list, dim_, result}, {out_size}};

    k.call(result);
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(cat_out, func, result)
}

Tensor transpose_hpu_lazy(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  PT_LAZY_TRACE;
  habana_lazy::SyncAccThreadPool();
  auto hl_self = GetHbLazyTensor(self);
  auto out = at::native::transpose(self, dim0_, dim1_);
  auto self_id = GetHbLazyTensor(self).getTensorUniqueId();
  auto out_id = GetHbLazyTensor(out).getTensorUniqueId();

  // at::native::transpose can return back self w/o invoking as_strided under
  // certain cases like 1D/dim0 == dim1. Skip view table access in such cases
  if ((out_id != self_id) && (is_fallback_original_op(self, out))) {
    auto hb_result = GetHbLazyTensor(out);
    {
      auto context = habana_lazy_executor.getDeviceExecutionContext(0);
      LOCK_VIEW_TABLE_MUTEX(context->viewContext);
      auto strided_param = HbLazyTensorViews::getViewTableParams(hb_result);
      strided_param->optype = kStridedOpTranspose;
      StridedOpTransposeParams transpose_param = {dim0_, dim1_};
      strided_param->params.transpose_param = transpose_param;

      PT_VIEWTABLE_DEBUG(
          "transpose fallback tensor id ",
          hl_self.getTensorUniqueId(),
          " dim0 ",
          dim0_,
          " dim1 ",
          dim1_);
    }
  }
  return out;
}

Tensor t_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  habana_lazy::SyncAccThreadPool();
  auto out = at::native::t(self);
  if (is_fallback_original_op(self, out)) {
    auto hb_result = GetHbLazyTensor(out);
    {
      auto context = habana_lazy_executor.getDeviceExecutionContext(0);
      LOCK_VIEW_TABLE_MUTEX(context->viewContext);
      auto strided_param = HbLazyTensorViews::getViewTableParams(hb_result);
      strided_param->optype = kStridedOpT;

      PT_VIEWTABLE_DEBUG(
          "t fallback tensor id ", GetHbLazyTensor(self).getTensorUniqueId());
    }
  }
  return out;
}

Tensor squeeze_hpu_lazy(const Tensor& self, int64_t dim_) {
  PT_LAZY_TRACE;

  auto dim = dim_;
  Tensor out;

  if (dim != HABANA_DIM_MAX) {
    dim = at::maybe_wrap_dim(dim_, self.dim());

    // no degenerate axis to squeeze
    if ((self.sizes()[dim] != 1) || (self.dim() == 1)) {
      return self;
    }

    out = at::native::squeeze(self, dim);
  } else {
    out = at::native::squeeze(self);
  }

  if (is_fallback_original_op(self, out)) {
    auto hb_result = GetHbLazyTensor(out);
    {
      auto context = habana_lazy_executor.getDeviceExecutionContext(0);
      LOCK_VIEW_TABLE_MUTEX(context->viewContext);
      auto strided_param = HbLazyTensorViews::getViewTableParams(hb_result);
      strided_param->optype = kStridedOpSqueeze;
      StridedOpSqueezeParams squeeze_param = {dim};
      strided_param->params.squeeze_param = squeeze_param;

      PT_VIEWTABLE_DEBUG(
          "squeeze fallback tensor id ",
          GetHbLazyTensor(self).getTensorUniqueId(),
          " dim ",
          dim);
    }
  }
  return out;
}

Tensor unsqueeze_hpu_lazy(const Tensor& self, int64_t dim_) {
  PT_LAZY_TRACE;

  auto dim = at::maybe_wrap_dim(dim_, self.dim() + 1);

  auto out = at::native::unsqueeze(self, dim);
  if (is_fallback_original_op(self, out)) {
    auto hb_result = GetHbLazyTensor(out);
    {
      auto context = habana_lazy_executor.getDeviceExecutionContext(0);
      LOCK_VIEW_TABLE_MUTEX(context->viewContext);
      auto strided_param = HbLazyTensorViews::getViewTableParams(hb_result);
      strided_param->optype = kStridedOpUnsqueeze;
      StridedOpSqueezeParams squeeze_param = {dim};
      strided_param->params.squeeze_param = squeeze_param;

      PT_VIEWTABLE_DEBUG(
          "unsqueeze fallback tensor id ",
          GetHbLazyTensor(self).getTensorUniqueId(),
          " dim ",
          dim);
    }
  }
  return out;
}

Tensor& unsqueeze_hpu_lazy_(Tensor& self, int64_t dim) {
  PT_LAZY_TRACE;

  return at::native::unsqueeze_(self, dim);
}

void adjustPTSizesLazy(Tensor& t) {
  // PT expects metadata like sizes and strides same as in NCHW,
  // but data permuted for channel last, so change the size and stride
  // NCHW
  auto sizes = t.sizes().vec();
  std::vector<int> out_pos = {
      LayoutFormatDims::N,
      LayoutFormatDims::W,
      LayoutFormatDims::C,
      LayoutFormatDims::H};
  std::vector<long int> swapped_sizes = {
      sizes[out_pos[0]],
      sizes[out_pos[1]],
      sizes[out_pos[2]],
      sizes[out_pos[3]]};
  std::vector<int> out_pos_5d = {
      LayoutFormatWithDepthDims::N,
      LayoutFormatWithDepthDims::W,
      LayoutFormatWithDepthDims::C,
      LayoutFormatWithDepthDims::D,
      LayoutFormatWithDepthDims::H};
  std::vector<long int> swapped_sizes_5d = {
      sizes[out_pos_5d[0]],
      sizes[out_pos_5d[1]],
      sizes[out_pos_5d[2]],
      sizes[out_pos_5d[3]],
      sizes[out_pos_5d[4]]};
  if (t.dim() == 5) {
    t.unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes_5d);
  } else {
    t.unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes);
  }
  // For 4D tensors we need to make sure that we generate the PT channel
  // last strides. Also as its a front end tensor, there may be a backend
  // tensor already if so, change dims for that tensor too.
  if (t.dim() == 4) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        c10::MemoryFormat::ChannelsLast);
    auto hl_result = GetHbLazyTensor(t);
    if (hl_result.getAttachedTensorImpl()) {
      hl_result.getAttachedTensorImpl()->empty_tensor_restride(
          c10::MemoryFormat::ChannelsLast);
    }
  }
  if (t.dim() == 5) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        c10::MemoryFormat::ChannelsLast3d);
    auto hl_result = GetHbLazyTensor(t);
    if (hl_result.getAttachedTensorImpl()) {
      hl_result.getAttachedTensorImpl()->empty_tensor_restride(
          c10::MemoryFormat::ChannelsLast3d);
    }
  }
}
Tensor permute_cl_hpu_lazy(const Tensor& self, IntArrayRef dims_in) {
  PT_LAZY_TRACE;
  auto dims_vec = dims_in.vec();
  for (unsigned i = 0; i < dims_in.size(); i++) {
    dims_vec[i] = at::maybe_wrap_dim(dims_in[i], self.dim(), true);
  }
  IntArrayRef dims_(dims_vec);

  std::vector<at::IValue> vector_of_inputs;

  vector_of_inputs = {self, dims_};

  using T = at::Tensor;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(const std::vector<at::IValue>& vector_of_inputs)
        : LazyOp<T>("hpu::permute_cl", vector_of_inputs, {}, {}, -1) {}

   private:
    T get_result_overrideable() override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      auto dims = inputs[1].toIntList();
      std::vector<int64_t> new_sizes, new_strides;
      std::tie(new_sizes, new_strides) =
          PermuteOperator::compute_output_shape(self, dims.vec());
      auto result =
          empty_strided_hpu_lazy(new_sizes, new_strides, self.options(), false);
      adjustPTSizesLazy(result);
      return result;
    }
  };

  Kernel kernel{vector_of_inputs};
  RUN_MAYBE_WITH_ACC_THREAD(permute_cl, kernel)
}

Tensor permute_hpu_lazy(const Tensor& self, IntArrayRef dims_in) {
  PT_LAZY_TRACE;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_PERMUTE_WITH_STRIDED_VIEW)) {
    auto hl_self = GetHbLazyTensor(self);
    auto out = at::native::permute(self, dims_in);
    if (is_fallback_original_op(self, out)) {
      auto hb_result = GetHbLazyTensor(out);
      {
        auto context = habana_lazy_executor.getDeviceExecutionContext(0);
        LOCK_VIEW_TABLE_MUTEX(context->viewContext);
        auto strided_param = HbLazyTensorViews::getViewTableParams(hb_result);
        strided_param->optype = kStridedOpPermute;
        strided_param->sizes = dims_in.vec();
        PT_VIEWTABLE_DEBUG(
            "permute fallback tensor id ",
            hl_self.getTensorUniqueId(),
            " dims_in ",
            dims_in.vec());
      }
    }
    return out;
  } else {
    auto dims_vec = dims_in.vec();
    for (unsigned i = 0; i < dims_in.size(); i++) {
      dims_vec[i] = at::maybe_wrap_dim(dims_in[i], self.dim(), true);
    }
    IntArrayRef dims_(dims_vec);

    std::vector<at::IValue> vector_of_inputs;

    vector_of_inputs = {self, dims_};

    using T = at::Tensor;
    class Kernel : public LazyOp<T> {
     public:
      Kernel(const std::vector<at::IValue>& vector_of_inputs)
          : LazyOp<T>("aten::permute", vector_of_inputs, {}, {}, -1) {}

     private:
      T get_result_overrideable() override {
        auto inputs = get_inputs();
        auto self = inputs[0].toTensor();
        auto dims = inputs[1].toIntList();
        std::vector<int64_t> new_sizes, new_strides;
        std::tie(new_sizes, new_strides) =
            PermuteOperator::compute_output_shape(self, dims.vec());
        auto result = empty_strided_hpu_lazy(
            new_sizes, new_strides, self.options(), false);
        return result;
      }
    };

    Kernel kernel{vector_of_inputs};
    return kernel.call();
  }
}

#if ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR < 13))
Tensor expand_hpu_lazy(const Tensor& self, IntArrayRef size_in, bool implicit) {
  PT_LAZY_TRACE;
#else
Tensor expand_hpu_lazy(const Tensor& self, SymIntArrayRef size, bool implicit) {
  PT_LAZY_TRACE;
  auto size_in = c10::asIntArrayRefSlow(size);
#endif
  // This ZST output tensor should ideally be handled at Synapse level, but
  // since it is throwing errors in that case we are forced to add this
  // work-around. E.g. self.sizes() = {1} size_in = {0}
  // TBD: Investigate and raise a JIRA on GC.
  auto size_vec = size_in.vec();
  auto flattened_size = std::accumulate(
      size_vec.begin(), size_vec.end(), 1, std::multiplies<int64_t>());

  if (flattened_size == 0) {
    auto result = empty_hpu_lazy(
        size_in.vec(), self.options(), self.suggest_memory_format(), true);
    auto hl_result = GetHbLazyTensor(result);
    updateDstDependencies(hl_result, result);
    flush_op(result);
    return result;
  }

  auto out = at::native::expand(self, size_in, implicit);
  auto hl_self = GetHbLazyTensor(self);
  auto hb_result = GetHbLazyTensor(out);
  auto self_id = hl_self.getTensorUniqueId();
  auto out_id = hb_result.getTensorUniqueId();

  if ((out_id != self_id) && (is_fallback_original_op(self, out))) {
    {
      auto context = habana_lazy_executor.getDeviceExecutionContext(0);
      LOCK_VIEW_TABLE_MUTEX(context->viewContext);
      auto strided_param = HbLazyTensorViews::getViewTableParams(hb_result);
      strided_param->optype = kStridedOpExpand;
      strided_param->sizes = size_in.vec();
      StridedOpExpandParams expand_param = {implicit};
      strided_param->params.expand_param = expand_param;
    }
    PT_VIEWTABLE_DEBUG(
        "expand fallback tensor id ",
        hl_self.getTensorUniqueId(),
        " sizes ",
        size_in.vec());
  }
  return out;
}

std::vector<Tensor> split_with_sizes_hpu_lazy(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  PT_LAZY_TRACE;
  // Changing the implementation of split with sizes
  // to follow the pytorch fork's approach of lowering
  // the split operation with multiple slice operations.
  // This avoids strided memcpy operations and uses
  // the SliceOp from GC which results in better perf

  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  int64_t cur_size = self.size(dim);
  int64_t num_splits = split_sizes.size();
  std::vector<Tensor> splits(num_splits);
  int64_t start_idx = 0;

  for (const auto i : c10::irange(num_splits)) {
    auto length = split_sizes[i];
    TORCH_CHECK(
        length >= 0,
        "split_with_sizes expects split_sizes have only non-negative ",
        "entries, but got split_sizes=",
        split_sizes);
    if (start_idx !=
        cur_size) { // start being the end is valid, but not a valid
      // dim specification.
      start_idx = c10::maybe_wrap_dim(start_idx, cur_size);
    }
    TORCH_CHECK(
        length >= 0 && start_idx <= cur_size - length,
        "start (",
        start_idx,
        ") + length (",
        length,
        ") exceeds dimension size (",
        cur_size,
        ").");
    splits[i] = slice_hpu_lazy(self, dim, start_idx, start_idx + length, 1);
    start_idx += length;
  }
  TORCH_CHECK(
      start_idx == cur_size,
      "split_with_sizes expects split_sizes to sum exactly to ",
      cur_size,
      " (input tensor's size at dimension ",
      dim,
      "), ",
      "but got split_sizes=",
      split_sizes);
  return splits;
};

std::tuple<Tensor, Tensor> topk_hpu_lazy_impl(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  PT_LAZY_TRACE;

  std::vector<at::IValue> vector_of_inputs;
  std::string op_name;
  std::set<size_t> metadata_indices;

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    op_name = "hpu::topk";
    auto k_tensor = empty_hpu_lazy(
        k, self.options(), self.suggest_memory_format(), false, SHAPE_TENSOR);
    vector_of_inputs = {self, k_tensor, dim, largest, sorted};
    metadata_indices = {2, 3, 4};
  } else {
    op_name = "aten::topk";
    vector_of_inputs = {self, k, dim, largest, sorted};
    metadata_indices = {1, 2, 3, 4};
  }

  using T = std::tuple<at::Tensor, at::Tensor>;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(
        const Tensor& self,
        int64_t k,
        int64_t dim,
        const std::string& op_name,
        const std::vector<at::IValue>& vector_of_inputs,
        std::set<size_t> metadata_indices)
        : LazyOp<T>(op_name, vector_of_inputs, metadata_indices, {}, -1),
          self(self),
          k(k),
          dim(dim) {}

   private:
    T get_result_overrideable() override {
      auto shape_out = self.sizes().vec();
      int64_t dim_ = c10::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
      shape_out[dim_] = k;
      auto type = kLong; // PyTorch expects returned indices dtype to be Long

      auto result_0 = empty_hpu_lazy(
          shape_out, self.options(), self.suggest_memory_format(), false);
      auto result_1 = empty_hpu_lazy(
          shape_out,
          self.options().dtype(type),
          self.suggest_memory_format(),
          false);
      return {result_0, result_1};
    }
    at::Tensor self;
    int64_t k;
    int64_t dim;
  };

  Kernel kernel{self, k, dim, op_name, vector_of_inputs, metadata_indices};
  return kernel.call();
}

std::tuple<Tensor&, Tensor&> topk_out_hpu_lazy_impl(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  PT_LAZY_TRACE;

  std::vector<at::IValue> vector_of_inputs;
  std::string op_name;
  std::set<size_t> metadata_indices;
  std::vector<std::vector<int64_t>> out_shapes;

  auto shape_out = self.sizes().vec();
  int64_t dim_ = c10::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  shape_out[dim_] = k;
  out_shapes = {shape_out, shape_out};

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    op_name = "hpu::topk";
    auto k_tensor = empty_hpu_lazy(
        k, self.options(), self.suggest_memory_format(), false, SHAPE_TENSOR);
    vector_of_inputs = {self, k_tensor, dim, largest, sorted, values, indices};
    metadata_indices = {2, 3, 4};
  } else {
    op_name = "aten::topk";
    vector_of_inputs = {self, k, dim, largest, sorted, values, indices};
    metadata_indices = {1, 2, 3, 4};
  }

  using T = std::tuple<at::Tensor&, at::Tensor&>;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(
        const Tensor& self,
        int64_t k,
        int64_t dim,
        Tensor& values,
        Tensor& indices,
        const std::string& op_name,
        const std::vector<at::IValue>& vector_of_inputs,
        std::set<size_t> metadata_indices,
        std::vector<std::vector<int64_t>> out_shapes)
        : LazyOp<T>(
              op_name,
              vector_of_inputs,
              metadata_indices,
              out_shapes,
              -1),
          self(self),
          k(k),
          dim(dim),
          values(values),
          indices(indices) {}

    at::Tensor self;
    int64_t k;
    int64_t dim;
    at::Tensor& values;
    at::Tensor& indices;
  };

  Kernel kernel{
      self,
      k,
      dim,
      values,
      indices,
      op_name,
      vector_of_inputs,
      metadata_indices,
      out_shapes};
  return kernel.call(std::tie(values, indices));
}

std::tuple<Tensor&, Tensor&> topk_out_hpu_lazy(
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  PT_LAZY_TRACE;
  topk_out_hpu_lazy_impl(self, k, dim_, largest, sorted, values, indices);
  return std::tie(values, indices);
}
// WA around for https://jira.habana-labs.com/browse/SW-57705
std::tuple<Tensor, Tensor> topk_hpu_lazy(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  PT_LAZY_TRACE;
  return topk_hpu_lazy_impl(self, k, dim, largest, sorted);
}

std::tuple<Tensor, Tensor> sort_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  PT_LAZY_TRACE;
  int64_t size_dim = self.dim() ? self.size(dim) : 1;
  dim = at::maybe_wrap_dim(dim, self.dim(), true);

  if (self.dim() == 0 && self.numel() == 1)
    return {self.clone(), at::zeros({}, TensorOptions(kHPU).dtype(at::kLong))};

  // Currently TPC supports only Axis 0(dim -1 in Pytorch) for topk.
  // For any other Axis, topk is called on the permuted input
  if (self.dim() > 0 && dim != self.dim() - 1) {
    std::vector<int64_t> permute_dims(self.dim());
    std::iota(permute_dims.begin(), permute_dims.end(), 0);
    std::swap(permute_dims[dim], permute_dims[self.dim() - 1]);
    auto permuted_self = permute_hpu_lazy(self, permute_dims);
    dim = self.dim() - 1;
    auto out =
        topk_hpu_lazy_impl(permuted_self, size_dim, dim, descending, true);
    auto permuted_out_0 = permute_hpu_lazy(std::get<0>(out), permute_dims);
    auto permuted_out_1 = permute_hpu_lazy(std::get<1>(out), permute_dims);
    return std::tie(permuted_out_0, permuted_out_1);
  } else
    return topk_hpu_lazy_impl(self, size_dim, dim, descending, true);
}

at::Tensor one_hot_hpu_lazy(const Tensor& self, int64_t num_classes) {
  PT_LAZY_TRACE;
  auto shape = self.sizes().vec();

  // empty tensor could be converted to one hot representation,
  // but shape inference is not possible.
  if (self.numel() == 0) {
    if (num_classes <= 0) {
      AT_ERROR("Can not infer total number of classes from empty tensor.");
    } else {
      shape.push_back(num_classes);
      return at::empty(shape, self.options());
    }
  }

  if (num_classes == -1) {
    num_classes = self.max().item().toLong() + 1;
  }

  LazyOp<at::Tensor> k(
      "aten::one_hot",
      {self, num_classes},
      {OneHotOperator::compute_output_shape(self, num_classes)});
  RUN_MAYBE_WITH_ACC_THREAD(one_hot, k)
}

Tensor upsample_nearest2d_hpu_lazy(
    const Tensor& input,
    OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_LAZY_TRACE;
  Tensor input_cast = input;
  if (input.scalar_type() == c10::ScalarType::Byte) {
    // u8 -> f32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {input, c10::ScalarType::Float},
        {input.sizes().vec()},
        c10::ScalarType::Float};
    input_cast = k_.call();
  }
  auto memory_format = input_cast.suggest_memory_format();
  LazyOp<at::Tensor> k(
      "aten::upsample_nearest2d",
      {input_cast, output_size, scale_factors},
      {1, 2},
      {UpsampleOperator::compute_output_shape(
          input_cast.sizes().vec(),
          output_size,
          scale_factors,
          memory_format)});
  auto result = k.call();
  if (input.scalar_type() == c10::ScalarType::Byte) {
    auto result_cast = result;
    // f32 -> i32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {result_cast, c10::ScalarType::Int},
        {result_cast.sizes().vec()},
        c10::ScalarType::Int};
    result_cast = k_.call();
    // i32 -> u8
    LazyOp<at::Tensor> k{
        "hpu::cast",
        {result_cast, input.scalar_type()},
        {result_cast.sizes().vec()},
        input.scalar_type()};
    result = k.call();
  }
  return result;
}

Tensor upsample_nearest2d_backward_hpu_lazy(
    const Tensor& grad_output,
    OptionalIntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_LAZY_TRACE;
  Tensor grad_output_cast = grad_output;
  if (grad_output.scalar_type() == c10::ScalarType::Byte) {
    // u8 -> f32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {grad_output, c10::ScalarType::Float},
        {grad_output.sizes().vec()},
        c10::ScalarType::Float};
    grad_output_cast = k_.call();
  }
  std::vector<int64_t> permuted_sizes = input_size.vec();
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) == false) {
    permuted_sizes[0] = input_size[0];
    permuted_sizes[1] = input_size[2];
    permuted_sizes[2] = input_size[3];
    permuted_sizes[3] = input_size[1];
  }

  std::string op;
  Stack args;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    op = "hpu::upsample_nearest2d_backward";
    auto input_shape = empty_hpu_lazy(
        permuted_sizes,
        grad_output.options(),
        grad_output.suggest_memory_format(),
        false,
        SHAPE_TENSOR);
    args = {grad_output_cast, output_size, input_shape, scale_factors};
  } else {
    op = "aten::upsample_nearest2d_backward";
    args = {grad_output_cast, output_size, permuted_sizes, scale_factors};
  }
  LazyOp<at::Tensor> k(op, args, {input_size.vec()});
  auto result = k.call();

  if (grad_output.scalar_type() == c10::ScalarType::Byte) {
    auto result_cast = result;
    // f32 -> i32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {result_cast, c10::ScalarType::Int},
        {result_cast.sizes().vec()},
        c10::ScalarType::Int};
    result_cast = k_.call();
    // i32 -> u8
    LazyOp<at::Tensor> k{
        "hpu::cast",
        {result_cast, grad_output.scalar_type()},
        {result_cast.sizes().vec()},
        grad_output.scalar_type()};
    result = k.call();
  }
  return result;
}

Tensor upsample_nearest3d_hpu_lazy(
    const Tensor& input,
    OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_LAZY_TRACE;
  Tensor input_cast = input;
  if (input.scalar_type() == c10::ScalarType::Byte) {
    // u8 -> f32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {input, c10::ScalarType::Float},
        {input.sizes().vec()},
        c10::ScalarType::Float};
    input_cast = k_.call();
  }
  auto memory_format = input_cast.suggest_memory_format();
  LazyOp<at::Tensor> k(
      "aten::upsample_nearest3d",
      {input_cast, output_size, scale_factors},
      {1, 2},
      {UpsampleOperator::compute_output_shape(
          input_cast.sizes().vec(),
          output_size,
          scale_factors,
          memory_format)});
  auto result = k.call();
  if (input.scalar_type() == c10::ScalarType::Byte) {
    auto result_cast = result;
    // f32 -> i32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {result_cast, c10::ScalarType::Int},
        {result_cast.sizes().vec()},
        c10::ScalarType::Int};
    result_cast = k_.call();
    // i32 -> u8
    LazyOp<at::Tensor> k{
        "hpu::cast",
        {result_cast, input.scalar_type()},
        {result_cast.sizes().vec()},
        input.scalar_type()};
    result = k.call();
  }
  return result;
}

Tensor upsample_nearest3d_backward_hpu_lazy(
    const Tensor& grad_output,
    OptionalIntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_LAZY_TRACE;
  Tensor grad_output_cast = grad_output;
  if (grad_output.scalar_type() == c10::ScalarType::Byte) {
    // u8 -> f32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {grad_output, c10::ScalarType::Float},
        {grad_output.sizes().vec()},
        c10::ScalarType::Float};
    grad_output_cast = k_.call();
  }
  std::vector<int64_t> permuted_sizes = input_size.vec();
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) == false) {
    permuted_sizes[0] = input_size[0];
    permuted_sizes[1] = input_size[2];
    permuted_sizes[2] = input_size[3];
    permuted_sizes[3] = input_size[4];
    permuted_sizes[4] = input_size[1];
  }
  LazyOp<at::Tensor> k(
      "aten::upsample_nearest3d_backward",
      {grad_output_cast, output_size, permuted_sizes, scale_factors},
      {1, 2, 3},
      {input_size.vec()});
  auto result = k.call();
  if (grad_output.scalar_type() == c10::ScalarType::Byte) {
    auto result_cast = result;
    // f32 -> i32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {result_cast, c10::ScalarType::Int},
        {result_cast.sizes().vec()},
        c10::ScalarType::Int};
    result_cast = k_.call();
    // i32 -> u8
    LazyOp<at::Tensor> k{
        "hpu::cast",
        {result_cast, grad_output.scalar_type()},
        {result_cast.sizes().vec()},
        grad_output.scalar_type()};
    result = k.call();
  }
  return result;
}

Tensor isfinite_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k_{
      "aten::isfinite", {input}, {input.sizes().vec()}, c10::ScalarType::Bool};
  RUN_MAYBE_WITH_ACC_THREAD(isfinite, k_)
}

Scalar _local_scalar_dense_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  habana_lazy::SyncAccThreadPool();
  Scalar out;
  // If self is a lazy tensor make sure the execution till the point of self
  // getting flled has finished before we start copying
  if (IsHbLazyTensor(self)) {
    HbLazyTensor hb_tensor = GetOrCreateHbLazyTensor(self, self.device());
    hb_tensor = HbLazyTensorViews::HandleViewsOrUpdate(self, hb_tensor);
    if (self.device().type() == c10::DeviceType::HPU) {
      flush_op({});
      // Trigger point execution
      PT_IRGRAPH_DEBUG("step marker due to local scalar");
      HbLazyTensor::StepMarker({});
    }
    // if there is a view, we need to sync before accessing the tensor_data.
    // This is because we skip view outputs in stepmarker
    hb_tensor = GetHbLazyTensor(HbLazyTensorViews::HandleViewsD2H(self));
    auto tensor_data = hb_tensor.GetHbLazyTensorData();
    out = habana_helpers::_local_scalar_dense_internal(tensor_data.value());
  } else {
    out = habana_helpers::_local_scalar_dense_internal(self);
  }
  return out;
}

Tensor fused_norm_hpu_lazy(
    std::vector<Tensor>& grad,
    const Tensor& max_norm,
    float norm_type) {
  PT_LAZY_TRACE;

  auto result = empty_hpu_lazy(
      {1}, grad[0].options(), grad[0].suggest_memory_format(), false);

  auto op_func = [grad, max_norm, norm_type, result]() mutable {
    bool is_view_evaluated = true;
    auto context = habana_lazy_executor.getDeviceExecutionContext(0);
    for (size_t i = 0; i < grad.size(); i++) {
      auto id = GetHbLazyTensor(grad[i]).getTensorUniqueId();
      StrideParams* params_ptr = context->viewContext.GetViewTableEntry(id);

      if (params_ptr != nullptr) {
        if (params_ptr->viewStatus != kEvaluated) {
          is_view_evaluated = false;
          break;
        }
      } else {
        is_view_evaluated = false;
        break;
      }
    }

    // grads are evaluated in case of gradient bucket view. Use the inplace
    // backend kernel to update same memory
    std::string node_str =
        is_view_evaluated ? "hpu::fused_norm_" : "hpu::fused_norm_lazy";

    ir::NodePtr node =
        std::make_shared<ir::FusedNorm>(grad, max_norm, norm_type, node_str);
    int64_t out_index = 0;

    auto hlgrad = habana_lazy::GetHbLazyTensor(grad[0]);
    habana_lazy::ir::Value& out1 = hlgrad.CurrentIrValue();
    node->set_as_output_tensor_list();
    out1.SetNode(
        node, hlgrad.GetDevice(), hlgrad.GetSizes(), hlgrad.dtype_optional());

    habana_lazy::ir::NodePtr node_unpack =
        std::make_shared<habana_lazy::ir::ListUnpack>(out1);

    auto hlresult = GetHbLazyTensor(result);
    ir::Value& out2 = hlresult.CurrentIrValue();
    out2.SetNode(
        node_unpack,
        hlresult.GetDevice(),
        hlresult.GetSizes(),
        hlresult.dtype_optional(),
        out_index++);

    // check if any of the grad is a view output and add strided insert node
    // accordingly
    for (size_t i = 0; i < grad.size(); i++) {
      auto grad_t = grad[i];
      auto hlgrad = GetHbLazyTensor(grad_t);
      auto id = hlgrad.getTensorUniqueId();
      StrideParams* params_ptr = context->viewContext.GetViewTableEntry(id);
      if ((params_ptr == nullptr) || (is_view_evaluated)) {
        ir::Value& out1 = hlgrad.CurrentIrValue();
        out1.SetNode(
            node_unpack,
            hlgrad.GetDevice(),
            hlgrad.GetSizes(),
            hlgrad.dtype_optional(),
            out_index++);
      } else {
        // fused norm has operated out of place on strided view's output
        auto clip_grad = empty_hpu_lazy(
            grad_t.sizes(),
            grad_t.options(),
            grad_t.suggest_memory_format(),
            false);
        auto hlgrad = GetHbLazyTensor(clip_grad);
        ir::Value& out1 = hlgrad.CurrentIrValue();
        out1.SetNode(
            node_unpack,
            hlgrad.GetDevice(),
            hlgrad.GetSizes(),
            hlgrad.dtype_optional(),
            out_index++);

        // add strided insert node. Do not flush in lazy eager as it is a fused
        // op. step marker will be used at the end
        strided_insert_hpu_lazy(grad_t, clip_grad, /*is_flush*/ false);
      }
    }

    flush_op(result);
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(fused_norm, op_func, result);
}

std::tuple<Tensor, Tensor> _unique_hpu_lazy(
    const Tensor& self,
    bool sorted,
    bool return_inverse) {
  PT_LAZY_TRACE;

  if (self.numel() == 0) {
    auto shape = DimVector{0};
    auto output = empty_hpu_lazy(
        shape, self.options(), self.suggest_memory_format(), true);
    auto hl_output = GetHbLazyTensor(output);
    updateDstDependencies(hl_output, output);
    flush_op(output);
    auto inverse_tensor = empty_hpu_lazy(
        shape,
        self.options().dtype(c10::ScalarType::Long),
        self.suggest_memory_format(),
        true);
    auto hl_inverse_tensor = GetHbLazyTensor(inverse_tensor);
    updateDstDependencies(hl_inverse_tensor, inverse_tensor);
    flush_op(inverse_tensor);
    return {output, inverse_tensor};
  }

  struct Unique : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> {
    explicit Unique(
        const std::vector<at::IValue>& inputs,
        const std::set<size_t>& metadata_indices = {},
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>>(
              "hpu::_unique",
              inputs,
              metadata_indices,
              out_shapes,
              -1) {}

    std::tuple<at::Tensor, at::Tensor, at::Tensor> get_result_overrideable()
        override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      int elements = self.numel();
      auto output_shape = at::DimVector{elements};
      auto valid_shape = at::DimVector{1};
      auto result0 = empty_hpu_lazy(
          output_shape, self.options(), self.suggest_memory_format(), false);
      auto result1 = empty_hpu_lazy(
          valid_shape,
          self.options().dtype(c10::ScalarType::Long),
          self.suggest_memory_format(),
          false);
      auto inverse_result = empty_hpu_lazy(
          output_shape,
          self.options().dtype(c10::ScalarType::Long),
          self.suggest_memory_format(),
          false);
      return {result0, result1, inverse_result};
    }
  };

  int elements = self.numel();
  std::vector<int64_t> feature_map_shape{elements};
  std::vector<int64_t> valid_count_shape{1};
  std::vector<int64_t> return_inverse_shape{elements};
  // Add unique node
  Unique k(
      {IValue(self), IValue(sorted), IValue(return_inverse)},
      {1, 2},
      {feature_map_shape, valid_count_shape, return_inverse_shape});
  // unique returns 2 output feature_map and valid tensor
  // and optional Inverse indices tensor and counts tensor
  auto output = k.call();
  auto feature_map = std::get<0>(output);
  auto valid_count = std::get<1>(output);
  // Force an execution here because "unique" is a non shape inferable op.
  // .item() internally triggers a mark_step
  PT_IRGRAPH_DEBUG("step marker due to unique");
  auto end = valid_count.item<int64_t>();
  StageSubmission::getInstance().setStageSubmissionFlow();

  // Add a slice node to capture relevent elements from feature_map
  auto result = slice_hpu_lazy(feature_map, 0, 0, end, 1);
  // Flipping to match the cpu results
  result = torch::flip(result, {0});
  habana_lazy::SyncAccThreadPool();
  flush_op(result);

  if (return_inverse) {
    auto inverse_tensor = std::get<2>(output);
    // Index flipping to match the cpu results
    Tensor subtracter = add_scalar_hpu_lazy(valid_count, 1, -1);
    auto inverse_result = add_tensor_hpu_lazy(subtracter, inverse_tensor, -1);
    habana_lazy::SyncAccThreadPool();
    inverse_result = view_hpu_lazy(inverse_result, self.sizes());

    flush_op(inverse_result);
    return std::make_tuple(result, inverse_result);
  } else {
    Tensor inverse_indices;
    return std::make_tuple(result, inverse_indices);
  }
};

std::tuple<Tensor, Tensor, Tensor> unique2_hpu_lazy(
    const Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  PT_LAZY_TRACE;
  if (self.numel() == 0) {
    auto result_ = empty_hpu_lazy(
        self.sizes(), self.options(), self.suggest_memory_format(), true);
    return std::make_tuple(result_, result_, result_);
  }
  struct Unique : LazyOp<std::tuple<at::Tensor, at::Tensor>> {
    explicit Unique(
        const std::vector<at::IValue>& inputs,
        const std::set<size_t>& metadata_indices = {},
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor>>(
              "hpu::_unique2",
              inputs,
              metadata_indices,
              out_shapes,
              -1) {}

    std::tuple<at::Tensor, at::Tensor> get_result_overrideable() override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      int elements = self.numel();
      auto output_shape = at::DimVector{elements};
      auto valid_shape = at::DimVector{1};
      auto result0 = empty_hpu_lazy(
          output_shape, self.options(), self.suggest_memory_format(), false);
      auto result1 = empty_hpu_lazy(
          valid_shape,
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format(),
          false);
      return {result0, result1};
    }
  };

  int elements = self.numel();
  std::vector<int64_t> feature_map_shape{elements};
  std::vector<int64_t> valid_count_shape{1};
  // Add unique_2 node
  Unique k(
      {IValue(self),
       IValue(sorted),
       IValue(return_inverse),
       IValue(return_counts)},
      {1, 2, 3},
      {feature_map_shape, valid_count_shape});
  // unique2 returns 2 output feature_map and valid tensor
  auto output = k.call();
  auto feature_map = std::get<0>(output);
  auto valid_count = std::get<1>(output);

  // Force an execution here because "unique" is a non shape inferable op.
  // .item() internally triggers a mark_step
  PT_IRGRAPH_DEBUG("step marker due to unique");
  auto end = valid_count.item<int64_t>();
  StageSubmission::getInstance().setStageSubmissionFlow();

  // Add a slice node to capture relevent elements from feature_map
  auto result = slice_hpu_lazy(feature_map, 0, 0, end, 1);

  // These are optional tensors which shall be populated only when we
  // start supporting return_inverse and return_counts
  Tensor inverse_indices;
  Tensor counts;
  flush_op(result);
  return std::make_tuple(result, inverse_indices, counts);
};

std::tuple<Tensor, Tensor, Tensor> unique_dim_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  PT_LAZY_TRACE;

  if (dim < 0) {
    dim = self.dim() + dim;
  }
  if (self.numel() == 0) {
    auto shape = DimVector{0};
    auto output = empty_hpu_lazy(
        shape, self.options(), self.suggest_memory_format(), true);
    auto hl_output = GetHbLazyTensor(output);
    updateDstDependencies(hl_output, output);
    flush_op(output);
    auto inverse_tensor = empty_hpu_lazy(
        shape,
        self.options().dtype(c10::ScalarType::Long),
        self.suggest_memory_format(),
        true);
    auto hl_inverse_tensor = GetHbLazyTensor(inverse_tensor);
    updateDstDependencies(hl_inverse_tensor, inverse_tensor);
    flush_op(inverse_tensor);
    auto counts_tensor = empty_hpu_lazy(
        shape,
        self.options().dtype(c10::ScalarType::Long),
        self.suggest_memory_format(),
        true);
    auto hl_counts_tensor = GetHbLazyTensor(counts_tensor);
    updateDstDependencies(hl_counts_tensor, counts_tensor);
    flush_op(counts_tensor);
    return {output, inverse_tensor, counts_tensor};
  }

  struct Unique
      : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>> {
    explicit Unique(
        const std::vector<at::IValue>& inputs,
        const std::set<size_t>& metadata_indices = {},
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>(
              "hpu::unique_dim",
              inputs,
              metadata_indices,
              out_shapes,
              -1) {}

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    get_result_overrideable() override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      auto dim = inputs[1].toInt();

      if (dim < 0) {
        dim = self.dim() + dim;
      }
      auto output_shape = at::DimVector(self.sizes());
      auto valid_shape = at::DimVector{1};
      auto inverse_tensor_shape = DimVector{self.sizes().vec().at(dim)};
      auto counts_tensor_shape = DimVector{self.sizes().vec().at(dim)};

      auto result0 = empty_hpu_lazy(
          output_shape, self.options(), self.suggest_memory_format(), false);
      auto result1 = empty_hpu_lazy(
          valid_shape,
          self.options().dtype(c10::ScalarType::Long),
          self.suggest_memory_format(),
          false);
      auto result2 = empty_hpu_lazy(
          inverse_tensor_shape,
          self.options().dtype(c10::ScalarType::Long),
          self.suggest_memory_format(),
          false);
      auto result3 = empty_hpu_lazy(
          counts_tensor_shape,
          self.options().dtype(c10::ScalarType::Long),
          self.suggest_memory_format(),
          false);
      return {result0, result1, result2, result3};
    }
  };

  std::vector<int64_t> feature_map_shape = self.sizes().vec();
  std::vector<int64_t> valid_count_shape{1};
  std::vector<int64_t> inverse_tensor_shape{feature_map_shape[dim]};
  std::vector<int64_t> counts_tensor_shape{feature_map_shape[dim]};
  // Add unique_dim node
  Unique k(
      {IValue(self),
       IValue(dim),
       IValue(sorted),
       IValue(return_inverse),
       IValue(return_counts)},
      {1, 2, 3, 4},
      {feature_map_shape,
       valid_count_shape,
       inverse_tensor_shape,
       counts_tensor_shape});
  // unique_dim returns 2 output feature_map and valid tensor
  auto output = k.call();
  auto feature_map = std::get<0>(output);
  auto valid_count = std::get<1>(output);
  auto inverse_tensor = std::get<2>(output);
  auto counts_tensor = std::get<3>(output);

  // Force an execution here because "unique" is a non shape inferable op.
  // .item() internally triggers a mark_step
  PT_IRGRAPH_DEBUG("step marker due to unique");
  auto end = valid_count.item<int64_t>();
  StageSubmission::getInstance().setStageSubmissionFlow();

  // Add a slice node to capture relevent elements from feature_map
  auto unique_result = slice_hpu_lazy(feature_map, dim, 0, end, 1);
  auto counts_result = slice_hpu_lazy(counts_tensor, 0, 0, end, 1);

  flush_op(unique_result);
  flush_op(inverse_tensor);
  flush_op(counts_result);

  if (return_inverse && return_counts) {
    return std::make_tuple(unique_result, inverse_tensor, counts_result);
  } else if (return_inverse && !return_counts) {
    Tensor counts;
    return std::make_tuple(unique_result, inverse_tensor, counts);
  } else if (!return_inverse && return_counts) {
    Tensor inverse;
    return std::make_tuple(unique_result, inverse, counts_result);
  } else {
    Tensor inverse;
    Tensor counts;
    return std::make_tuple(unique_result, inverse, counts);
  }
};

std::tuple<at::Tensor, at::Tensor> max_dim_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  habana_lazy::SyncAccThreadPool();
  PT_LAZY_TRACE;
  std::vector<at::IValue> vector_of_inputs;
  vector_of_inputs = {self, dim, keepdim};

  using T = ::std::tuple<at::Tensor, at::Tensor>;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(const std::vector<at::IValue>& vector_of_inputs)
        : LazyOp<T>("hpu::max_dim", vector_of_inputs, {}, {}, -1) {}

   private:
    T get_result_overrideable() override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      auto dim = inputs[1].toInt();
      auto keepdim = inputs[2].toBool();

      auto shape = ReduceOperator::compute_output_shape(self, dim, keepdim);
      auto values = empty_hpu_lazy(
          shape, self.options(), self.suggest_memory_format(), false);
      auto indices = empty_hpu_lazy(
          shape,
          self.options().dtype(c10::ScalarType::Long),
          self.suggest_memory_format(),
          false);
      return T(values, indices);
    }
  };

  Kernel kernel{vector_of_inputs};
  return kernel.call();
}

at::Tensor max_hpu_lazy(const at::Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::max", {self}, {}, {{}}};
  RUN_MAYBE_WITH_ACC_THREAD(max, k)
}

at::Tensor min_hpu_lazy(const at::Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::min", {self}, {}, {{}}};
  RUN_MAYBE_WITH_ACC_THREAD(min, k)
}

Tensor masked_scale_hpu_lazy(
    const Tensor& self,
    const Tensor& mask,
    double scale) {
  PT_LAZY_TRACE;
  // scale changed to support dropout backward based on what we pass for
  // dropout
  scale = scale / (scale - 1);
  auto masked = torch::mul(self, mask);
  auto scaled = torch::mul(masked, scale);
  return scaled;
}

Tensor matmul_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyOp<Tensor> k(
      "aten::matmul",
      {self, other},
      {},
      {MatMulOperator::compute_output_shape(self, other)});
  RUN_MAYBE_WITH_ACC_THREAD(matmul, k)
}

std::tuple<Tensor, Tensor> matmul_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;
  LazyOp<std::tuple<Tensor, Tensor>> k(
      "hpu::matmul_backward",
      {grad_output, self, other},
      {},
      {self.sizes().vec(), other.sizes().vec()});
  return k.call();
}

Tensor habana_nms_hpu_lazy(
    const Tensor& boxes,
    const Tensor& scores,
    float iou_threshold,
    float score_threshold) {
  PT_LAZY_TRACE;
  habana_lazy::SyncAccThreadPool();
  // Ensuring that the boxes and scores input to nms is always FP32
  // This is required because when batched_nms is called
  // it calls torch.ops.torhchvision.nms which is visible
  // only internally through C++ flow. Instead of changing
  // the call in torchvision to call torchvision.ops.nms(external
  // facing API exposed via python), we chose to handle it
  // internally in the bridge to ensure boxes input to NMS
  // is always FP32. Also the topk operation does
  // not run on BF16 which is used by NMS internally
  // Consider removing this FP32 restriction once complex guid
  // implementation for NMS is in place

  Tensor boxes_cast = boxes;
  Tensor scores_cast = scores;
  if (boxes.scalar_type() == c10::ScalarType::BFloat16) {
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {boxes, c10::ScalarType::Float},
        {boxes.sizes().vec()},
        c10::ScalarType::Float};
    boxes_cast = k_.call();
  }
  if (scores.scalar_type() == c10::ScalarType::BFloat16) {
    LazyOp<at::Tensor> s_{
        "hpu::cast",
        {scores, c10::ScalarType::Float},
        {scores.sizes().vec()},
        c10::ScalarType::Float};
    scores_cast = s_.call();
  }

  struct HabanaNMSLazy
      : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> {
   public:
    explicit HabanaNMSLazy(
        const std::vector<at::IValue>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>>(
              "hpu::habana_nms",
              inputs,
              {},
              out_shapes,
              -1) {}

    std::tuple<at::Tensor, at::Tensor, at::Tensor> get_result_overrideable()
        override {
      std::tuple<at::Tensor, at::Tensor, at::Tensor> results;
      auto inputs = get_inputs();
      auto scores = inputs[1].toTensor();
      auto box_id_out_shape = get_out_shapes()[0];
      auto valid_box_id_out_shape = get_out_shapes()[1];
      auto shape_tensor_shape = get_out_shapes()[2];
      std::get<0>(results) = empty_hpu_lazy(
          box_id_out_shape,
          scores.options().dtype(c10::ScalarType::Long),
          scores.suggest_memory_format(),
          false);
      std::get<1>(results) = empty_hpu_lazy(
          valid_box_id_out_shape,
          scores.options().dtype(c10::ScalarType::Long),
          scores.suggest_memory_format(),
          false);
      std::get<2>(results) = empty_hpu_lazy(
          shape_tensor_shape,
          scores.options().dtype(c10::ScalarType::Long),
          scores.suggest_memory_format(),
          false);
      return results;
    }
  };

  std::vector<int64_t> box_id_out_shape{scores.sizes()[0]};
  std::vector<int64_t> valid_box_id_out_shape{1};
  std::vector<int64_t> shape_tensor_shape{5};
  HabanaNMSLazy k(
      {boxes_cast, scores_cast, Scalar(iou_threshold), Scalar(score_threshold)},
      {box_id_out_shape, valid_box_id_out_shape, shape_tensor_shape});
  auto result_nms = k.call();
  auto box_id_out = std::get<0>(result_nms);
  auto valid_box_id_out = std::get<1>(result_nms);
  auto shape_tensor = std::get<2>(result_nms);

  // Force an execution here to capture valid_box_id_out.
  // This element is required to determine shape of next node's output
  PT_IRGRAPH_DEBUG("step marker due to nms");
  // .item() internally triggers a mark_step
  auto end = valid_box_id_out.item<int64_t>();
  StageSubmission::getInstance().setStageSubmissionFlow();

  // Extract correct output using shape information.
  // Add a slice node to capture relevent elements
  auto result = slice_hpu_lazy(box_id_out, 0, 0, end, 1);
  flush_op(result);
  return result;
}

Tensor batched_nms_hpu_lazy(
    const Tensor& boxes,
    const Tensor& scores,
    const Tensor& indexes,
    float iou_threshold) {
  PT_LAZY_TRACE;
  habana_lazy::SyncAccThreadPool();

  if (boxes.numel() == 0 && scores.numel() == 0 && indexes.numel() == 0) {
    auto shape = DimVector{0};
    auto output = empty_hpu_lazy(
        shape,
        scores.options().dtype(c10::ScalarType::Long),
        scores.suggest_memory_format(),
        true);
    auto hl_output = GetHbLazyTensor(output);
    updateDstDependencies(hl_output, output);
    flush_op(output);
    return output;
  }
  // Ensuring that the boxes and scores input to batched_nms is always FP32,
  // this is because CGUID expects boxes to be f32. TBD: move this cast
  // addition to HMP
  Tensor boxes_cast = boxes;
  Tensor scores_cast = scores;
  if (boxes.scalar_type() == c10::ScalarType::BFloat16) {
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {boxes, c10::ScalarType::Float},
        {boxes.sizes().vec()},
        c10::ScalarType::Float};
    boxes_cast = k_.call();
  }
  if (scores.scalar_type() == c10::ScalarType::BFloat16) {
    LazyOp<at::Tensor> s_{
        "hpu::cast",
        {scores, c10::ScalarType::Float},
        {scores.sizes().vec()},
        c10::ScalarType::Float};
    scores_cast = s_.call();
  }

  struct BatchedNMSLazy : LazyOp<std::tuple<at::Tensor, at::Tensor>> {
   public:
    explicit BatchedNMSLazy(
        const std::vector<at::IValue>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor>>(
              "hpu::batched_nms",
              inputs,
              {},
              out_shapes,
              -1) {}

    std::tuple<at::Tensor, at::Tensor> get_result_overrideable() override {
      std::tuple<at::Tensor, at::Tensor> results;
      auto inputs = get_inputs();
      auto scores = inputs[1].toTensor();
      auto box_id_out_shape = get_out_shapes()[0];
      auto shape_tensor_shape = get_out_shapes()[1];
      std::get<0>(results) = empty_hpu_lazy(
          box_id_out_shape,
          scores.options().dtype(c10::ScalarType::Long),
          scores.suggest_memory_format(),
          false);
      std::get<1>(results) = empty_hpu_lazy(
          shape_tensor_shape,
          scores.options().dtype(c10::ScalarType::Int),
          scores.suggest_memory_format(),
          false);
      return results;
    }
  };

  // max_classes set for COCO dataset for now, can be increased in future
  // based on requirement. larger max_classes => smaller max size for
  // num_boxes allowed because of memory trade-off.
  constexpr int max_classes = 81;

  std::vector<int64_t> box_id_out_shape{scores.sizes()[0] * max_classes};
  std::vector<int64_t> shape_tensor_shape{5};
  auto shape_tensor_1 = empty_hpu_lazy(
      scores.sizes(),
      indexes.options().dtype(c10::ScalarType::Int),
      c10::MemoryFormat::Contiguous,
      false,
      SHAPE_TENSOR);

  auto shape_tensor_2 = empty_hpu_lazy(
      {scores.sizes()[0] * max_classes},
      indexes.options().dtype(c10::ScalarType::Int),
      c10::MemoryFormat::Contiguous,
      false,
      SHAPE_TENSOR);

  BatchedNMSLazy k(
      {boxes_cast,
       scores_cast,
       indexes,
       Scalar(iou_threshold),
       shape_tensor_1,
       shape_tensor_2,
       Scalar(max_classes)},
      {box_id_out_shape, shape_tensor_shape});
  auto result_nms = k.call();
  auto box_id_out = std::get<0>(result_nms);
  auto shape_tensor = std::get<1>(result_nms);

  // Force an execution here to capture valid_box_id_out.
  // This element is required to determine shape of next node's output
  PT_IRGRAPH_DEBUG("step marker due to nms");
  // .item() internally triggers a mark_step
  auto end = shape_tensor[0].item<int64_t>();

  // Extract correct output using shape information.
  // Add a slice node to capture relevent elements
  auto result = slice_hpu_lazy(box_id_out, 0, 0, end, 1);
  flush_op(result);
  return result;
}

at::Tensor roi_align_fwd_hpu_lazy(
    const at::Tensor& images,
    const at::Tensor& rois,
    const at::Tensor& num_rois,
    int output_h,
    int output_w,
    int mode,
    int sampling_ratio,
    float spatial_scale,
    bool aligned) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  std::shared_ptr<LazyOp<Tensor>> cast_op_ptr;

  // Assuming outshape to be NCHW
  std::vector<int64_t> out_shape{
      num_rois.sizes()[0], images.sizes()[1], output_h, output_w};
  auto rois_f32 = rois;
  // TPC expects rois to be always fp32, therefore adding this cast
  if (rois.scalar_type() == c10::ScalarType::BFloat16) {
    // Cast temp tensor to orig_out tensor data type
    cast_op_ptr = std::make_shared<LazyOp<Tensor>>(LazyOp<Tensor>{
        "hpu::cast", {rois, c10::ScalarType::Float}, {}, {rois.sizes().vec()}});
    rois_f32 = cast_op_ptr.get()->get_result();
  }
  LazyOp<at::Tensor> k(
      "hpu::roi_align_fwd",
      {images,
       rois_f32,
       num_rois,
       output_h,
       output_w,
       mode,
       sampling_ratio,
       spatial_scale,
       aligned},
      {},
      {out_shape});
  auto out = k.get_result();

  auto func = [op = std::move(k),
               cast_op_ptr = std::move(cast_op_ptr),
               rois_f32,
               out]() mutable {
    if (cast_op_ptr) {
      cast_op_ptr.get()->call(rois_f32);
    }
    op.call(out);
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(roi_align_fwd, func, out)
}

at::Tensor roi_align_bwd_hpu_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& rois,
    const at::Tensor& num_rois,
    int bs,
    int ch,
    int h,
    int w,
    int sampling_ratio,
    float spatial_scale,
    bool aligned) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  std::vector<int64_t> out_shape = {bs, ch, h, w};
  auto input_shape = empty_hpu_lazy(
      out_shape, grad_out.options(), grad_out.suggest_memory_format(), true);
  LazyOp<at::Tensor> k(
      "hpu::roi_align_bwd",
      {grad_out,
       rois,
       num_rois,
       input_shape,
       sampling_ratio,
       spatial_scale,
       aligned},
      {},
      {out_shape});
  RUN_MAYBE_WITH_ACC_THREAD(roi_align_bwd, k)
}

Tensor silu_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k("aten::silu", {self});
  return k.call();
}

Tensor& silu_out_hpu_lazy(const Tensor& self, Tensor& out) {
  PT_LAZY_TRACE;
  CONVERT_0D_TO_1D(self)
  LazyOp<at::Tensor&> k("aten::silu", {out, self});
  auto& result = k.call(out);
  CONVERT_1D_TO_0D(self, result)
  return result;
}

Tensor cumsum_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  PT_LAZY_TRACE;
  at::Tensor self_updated_dtype = self;
  if (dtype.has_value() && (dtype.value() != self.scalar_type())) {
    self_updated_dtype = self.to(dtype.value());
  }

  LazyOp<at::Tensor> k{"aten::cumsum", {self_updated_dtype, dim, dtype}};
  return k.call();
}

at::Tensor& broadcast_hpu_lazy_(
    at::Tensor& tensor,
    int64_t root_rank,
    int64_t comm_id) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k(
      "hccl::broadcast_", {tensor, root_rank, comm_id}, {1, 2}, {}, 0);
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(broadcast_, k, tensor)
}

at::Tensor& allreduce_hpu_lazy_(
    at::Tensor& tensor,
    uint8_t reduce_op,
    int64_t comm_id) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k(
      "hccl::allreduce_", {tensor, reduce_op, comm_id}, {1, 2}, {}, 0);
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(allreduce_, k, tensor)
}

at::Tensor& reduce_hpu_lazy_(
    at::Tensor& tensor,
    int64_t dst_rank,
    uint8_t reduce_op,
    int64_t comm_id) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k(
      "hccl::reduce_",
      {tensor, dst_rank, reduce_op, comm_id},
      {1, 2, 3},
      {},
      0);
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(reduce_, k, tensor)
}

at::Tensor& alltoall_hpu_lazy_out(
    const at::Tensor& inputTensor,
    int64_t comm_id,
    at::Tensor& outputTensor) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k(
      "hccl::alltoall_out", {inputTensor, comm_id, outputTensor}, {1}, {}, 2);
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(alltoall_out, k, outputTensor)
}

at::Tensor& allgather_hpu_lazy_out(
    const at::Tensor& inputTensor,
    int64_t comm_id,
    at::Tensor& outputTensor) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k(
      "hccl::allgather_out", {inputTensor, comm_id, outputTensor}, {1}, {}, 2);
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(allgather_out, k, outputTensor)
}

at::Tensor& reduce_scatter_hpu_lazy_out(
    const at::Tensor& inputTensor,
    uint8_t reduce_op,
    int64_t comm_id,
    at::Tensor& outputTensor) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k(
      "hccl::reduce_scatter_out",
      {inputTensor, reduce_op, comm_id, outputTensor},
      {1, 2},
      {},
      3);
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(reduce_scatter_out, k, outputTensor)
}

at::Tensor& send_hpu_lazy_(
    at::Tensor& tensor,
    int64_t dst_rank,
    int64_t tag,
    int64_t comm_id) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k(
      "hccl::send_", {tensor, dst_rank, tag, comm_id}, {1, 2, 3}, {}, 0);
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(send_, k, tensor)
}

at::Tensor& recv_hpu_lazy_(
    at::Tensor& tensor,
    int64_t src_rank,
    int64_t tag,
    int64_t comm_id) {
  PT_LAZY_TRACE;
  habana_lazy::SyncAccThreadPool();
  LazyOp<at::Tensor&> k(
      "hccl::recv_", {tensor, src_rank, tag, comm_id}, {1, 2, 3}, {}, 0);
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(recv_, k, tensor)
}

Tensor linear_non2d_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  PT_LAZY_TRACE;
  /* Implements:
    auto output = at::matmul(input, weight.t());
    if (bias->defined()) {
      output.add_(*bias);
    }
    return output;
  */
  auto sizes = habana::MatMulOperator::compute_output_shape(
      input, weight, true /*weight transposed*/);
  LazyOp<at::Tensor> k("aten::linear", {input, weight, bias_opt}, {}, {sizes});
  RUN_MAYBE_WITH_ACC_THREAD(linear, k)
}

std::vector<at::Tensor> linear_non2d_bwd_hpu_lazy(
    const at::Tensor grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {
  PT_LAZY_TRACE;
  /*
  Implements:
    std::tuple<Tensor, Tensor> result;
    variable_list saved_vars = ctx->get_saved_variables();
    Tensor bias_grad;
    Tensor weight_grad;
    Tensor weight_trnsp = saved_vars[1].t();

    result =
        matmul_backward_hpu_lazy(grad_output[0], saved_vars[0], weight_trnsp);

    if (saved_vars[2].defined()) {
      bias_grad = grad_output[0].sum(0);
    }

    weight_grad = std::get<1>(result).t();
    return {std::get<0>(result), weight_grad, bias_grad};
  */

  auto bias_elem_count =
      bias_opt.value_or(Tensor()).defined() ? weight.sizes().vec()[0] : 0;
  std::vector<int64_t> bias_grad_sizes(1, bias_elem_count);
  LazyOp<std::tuple<Tensor, Tensor, Tensor>> k(
      "hpu::linear_non2d_bwd",
      {grad_output, input, weight, bias_opt},
      {},
      {input.sizes().vec(), weight.sizes().vec(), bias_grad_sizes});
  auto res = k.call();
  std::vector<at::Tensor> res_vec;
  res_vec.emplace_back(std::get<0>(res));
  res_vec.emplace_back(std::get<1>(res));
  if (bias_opt.value_or(Tensor()).defined()) {
    res_vec.emplace_back(std::get<02>(res));
  } else {
    res_vec.emplace_back(Tensor());
  }
  return res_vec;
}

at::Tensor habana_cast_to_fp8_lazy(
    const at::Tensor& input,
    bool stochastic_rounding,
    int seed) {
  PT_OP_TRACE;
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k_{
      "hpu::habana_cast_sr_mode",
      {input, c10::ScalarType::Fp8r152, stochastic_rounding, seed},
      {input.sizes().vec()},
      c10::ScalarType::Fp8r152};
  RUN_MAYBE_WITH_ACC_THREAD(cast_to_fp8, k_)
}

} // namespace habana_lazy
