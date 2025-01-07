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
#include <ATen/InferSize.h>
#include <ATen/native/TypeProperties.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch_ver/csrc/distributed/c10d/Types.hpp>
#include <cstdlib>
#include <ctime>
#include <type_traits>
#include <utility>
#include "backend/backend_meta.h"
#include "backend/habana_device/HPUAllocator.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/random.h"
#include "backend/synapse_helpers/device_helpers.h"
#include "common/dump_args.h"
#include "habana_helpers/frontend_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/embedding_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_kernels/loss_kernels.h"
#include "habana_kernels/nonzero_kernel.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/repeat.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/lazy_graph_hash_builder.h"
#include "habana_lazy/ops/cast_ops.h"
#include "habana_lazy/ops/embedding_bag.h"
#include "habana_lazy/ops/index.h"
#include "habana_lazy/ops/norm.h"
#include "habana_lazy/ops/shape_ops.h"
#include "habana_lazy/ops/unpack.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/sbs_debug.h"
#include "habana_lazy/view_utils.h"
#include "hpu_ops/bincount.h"
#include "hpu_ops/ctc_loss_custom.h"
#include "hpu_ops/fp8_ops.h"
#include "hpu_ops/masked_batch_gemm.h"
#include "hpu_ops/op_logger.h"
#include "hpu_ops/optimizer_lamb_gen.h"
#include "hpu_ops/sdpa_gen.h"
#include "lazy_kernels_declarations.h"
#include "lazy_optimizer_kernels.h"
#include "pytorch_helpers/habana_helpers/dtype_helpers.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

#define MAX_DIMS_FOR_ADVANCED_INDEXING (8)

using namespace habana;
using namespace at;

#define FP8_CHECK                                 \
  TORCH_CHECK(                                    \
      synapse_helpers::device_supports_fp8(       \
          HPUDeviceContext::get_device().type()), \
      "FP8 data type is not available on this device.")

namespace {
void AddMemcpy(const Tensor& src, Tensor& dst) {
  using namespace habana_lazy;
  // propagate constant tensor status through copy
  habana::get_and_set_tensor_const(src, dst);

  if ((src.numel() < dst.numel()) || (src.dim() < dst.dim())) {
    // using crude check of numel() to determine broadcast scenario to avoid
    // increase in host time
    auto t = dst;
    auto t_opt = c10::make_optional(t);
    HbLazyTensorViews::add_expand_lazy(
        src, dst.sizes().vec(), false /*implicit*/, t_opt);
    return;
  }

  auto hl_dst = GetOrCreateHbLazyTensor(dst);
  auto hl_src = GetHbLazyTensor(src);
  hl_src.SetOpAccumulationInProgress();
  hl_dst.SetOpAccumulationInProgress();

  // add control edge to avoid GC error " writing to already
  // registered graph output"

  // Add a control edge for habana_d2d_memcpy_other second input
  // as it may cause wrong order of execution, as shown below -
  //   z = add(x, i)
  //   x' = habana_d2d_memcpy_other(y, x)
  // Here, x is an input, but habana_d2d_memcpy_other actually updates
  // x and hence should come after add with a control edge
  updateDstDependencies(dst);

  auto copy_node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::habana_d2d_memcpy_other"),
      {hl_src.GetIrValue(), hl_dst.GetIrValue()});

  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::get_device_lazy_execution_context();
  context->RegisterTensor(hl_dst.getDataPtr());
  hl_dst.IrSetNode(copy_node);
  std::vector<at::Tensor> input_pt_vec;
  input_pt_vec.push_back(src);
  input_pt_vec.push_back(dst);
  copy_node->AddInputPtTensors(input_pt_vec);
  flush_op(1);
}
constexpr size_t INPUT_BATCH_INDEX = 0;
constexpr size_t INPUT_CHANNEL_INDEX = 1;

} // namespace

namespace habana_lazy {
const std::vector<int64_t> device_shape_tensor_size = {SYN_MAX_TENSOR_DIM};

bool is_inplace(at::Symbol symbol) {
  auto node_name = symbol.toQualString();
  /*
  Since as_strided_lazy is now out of place op, we need a control edge to create
  new tensor for fill to avoid GC error
  %5 : Float(*, requires_grad=0,
  device=hpu:0) = hpu::as_strided_lazy(%id:3, %2, %3, %4)
  %6 : Float(*, requires_grad=0, device=hpu:0) = aten::fill_(%5, %1)
  */

  const static std::unordered_set<std::string>
      underscored_ops_reported_as_non_inplace = {
          "aten::fill_",
          "hpu::uniform_",
          "hpu::random_",
          "hpu::normal_",
          "hpu::geometric_",
          "aten::zero_",
          "hpu::bernoulli_",
          "hpu::exponential_",
          "hpu::log_normal_"};
  if (underscored_ops_reported_as_non_inplace.find({node_name}) !=
      underscored_ops_reported_as_non_inplace.end()) {
    return false;
  }

  size_t len = strlen(node_name);
  char endch = node_name[len - 1];
  return endch == '_';
}

namespace {
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
    PT_IRGRAPH_DEBUG("step marker due to flushWithMarkStep");
    HbLazyTensor::StepMarker({});
  }
}
} // namespace

// For the ops that don't use LazyOp to construct nodes.
// Remove when all ops move to LazyOp style.
void flush_op(
    size_t out_tensor_count,
    std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info,
    std::vector<HbLazyTensor> out_hb_lazy_tensor) {
  // Count number of ops added by both accumulation thread and the main thread.
  // This is not accurate number of ops. The accurate number of ops can be taken
  // from accumulated ops (incrementAccumulatedOps).
  StageSubmission::getInstance().incrementOpCount();
  if (habana_lazy::AccThread::IsAccThreadEnabled() &&
      habana_lazy::AccThread::Get().inAccThreadContext()) {
    // Early exit. Ensure that StepMarker is not called from the accumulation
    // thread. StepMarker can deallocate tensors and it can cause deadlock.
    return;
  }

  const bool m_flush_op = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2;
  const bool m_random_flush = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 3;
  StageSubmission::getInstance().incrementAccumulatedOps();
  SBSDebug::getInstance().IncreaseOpsAndTensors(out_tensor_count);

  if (m_flush_op) {
    bool async =
        (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD) &&
         GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_EAGER_EXECUTION_THREAD));
    PT_IRGRAPH_DEBUG("step marker due to flush_op");
    HbLazyTensor::StepMarker(
        {}, lazy_front_end_info, out_hb_lazy_tensor, async);
  } else if (m_random_flush) {
    flushWithMarkStep();
  } else if (StageSubmission::getInstance().isExceededMaxAccumlatedSize()) {
    PT_LAZY_DEBUG("Reached max accumulated graph size, triggering a mark_step");
    PT_IRGRAPH_DEBUG("step marker due to max accumulated graph size");
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
  if (habana_helpers::is_downcast_to_int_needed(src.scalar_type())) {
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
    if (habana_helpers::is_downcast_to_int_needed(dst.scalar_type()) ||
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

void updateDstDependencies(const Tensor& dst) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    return;
  };

  auto hb_result = GetHbLazyTensor(dst);
  auto node = ir::Node::Create(
      Symbol::fromQualString("hpu::control_edge_"), {hb_result.GetIrValue()});
  node->set_as_control_edge();
  std::vector<at::Tensor> input_pt_vec;
  input_pt_vec.push_back(dst);
  hb_result.IrSetNode(node);
  node->AddInputPtTensors(input_pt_vec);
  RUNNING_HASH_COMBINE_OPERATOR(hpu::control_edge_, {dst});
}

Tensor _copy_from_and_resize(const Tensor& self, const Tensor& dst) {
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
  auto& stride_params_opt =
      GetHbLazyTensor(self, true, true).getDataPtr()->stride_params;
  TORCH_CHECK(stride_params_opt.has_value(), "incorrect tensor id");
  StrideParams& params = stride_params_opt.value();

  if (params.optype == kStridedOpDefault) {
    params.viewStatus = kViewWrite;
    params.write_cnt++;
  }

  // pick the most recent version
  Tensor recent_orig_t = HbLazyTensorViews::get_recent_base_tensor(params.base);
  auto recent_insert_t = HbLazyTensorViews::get_recent_base_tensor(insert_t);
  auto back_to_back_slices = HbLazyTensorViews::getSliceInsertParams(
      recent_orig_t, recent_insert_t, params);

  at::Tensor out;
  if (back_to_back_slices.empty()) {
    out = add_strided_insert_node(
        recent_orig_t,
        recent_insert_t,
        params.strides,
        params.offset,
        is_flush);
  } else {
    out = add_slice_insert_node(
        recent_orig_t, recent_insert_t, back_to_back_slices);
  }

  // update orig tensor map
  GetHbLazyTensor(params.base, true, false).getDataPtr()->recent_base = out;
  GetHbLazyTensor(recent_orig_t, true, false).SetOpAccumulationInProgress();
  GetHbLazyTensor(self, true, false).SetOpAccumulationInProgress();
  GetHbLazyTensor(self, false, false).SetOpAccumulationInProgress();

  PT_VIEWTABLE_DEBUG(
      "orig tensor map entry created for ", GetHbLazyTensorId(params.base));
  return;
}

/* checks if fallback to original op is possible*/
bool is_fallback_original_op(const Tensor& self) {
  PT_LAZY_TRACE;
  bool is_fallback = true;

  // trace until the base tensor is reached and check if there are any
  // as_strided ops fall back not possible if there are as_strided ops in the
  // sequence.
  auto stride_params_opt = GetHbLazyTensor(self).getDataPtr()->stride_params;
  while (stride_params_opt.has_value()) {
    if (stride_params_opt.value().optype == kStridedOpDefault) {
      is_fallback = false;
      break;
    }

    auto hl_parent = GetHbLazyTensor(stride_params_opt.value().parent);
    stride_params_opt = hl_parent.getDataPtr()->stride_params;
  }
  return is_fallback;
}

void lazy_view_fallback_handle(
    const Tensor& self,
    const Tensor& out,
    std::function<void(const Tensor&, StrideParams&)> func,
    std::function<bool(const Tensor&, const Tensor&)> additional_predicate =
        [](const Tensor&, const Tensor&) { return true; }) {
  PT_LAZY_TRACE;
  if (additional_predicate(self, out) && is_fallback_original_op(self)) {
    auto& strided_param_opt = GetHbLazyTensor(out).getDataPtr()->stride_params;
    TORCH_CHECK(strided_param_opt.has_value(), "invalid stride params");

    func(self, strided_param_opt.value());
  }
}

at::Tensor append_to_batch_h2d_list(const at::Tensor& scalar_tensor) {
  const auto& t =
      empty_hpu_lazy({}, scalar_tensor.options(), c10::nullopt, true);

  t.unsafeGetTensorImpl()->set_wrapped_number(true);

  auto func = [scalar_tensor, t]() {
    const auto& context = get_device_lazy_execution_context();

    bool processed = false;
    const auto& tensor = preProcessIfLongorDouble(scalar_tensor, t, processed);

    // Mark as input
    HbLazyTensor hb_tensor = GetOrCreateHbLazyTensor(t);
    hb_tensor.IrInitAsInputNode();
    context->MarkTensorStatus(
        hb_tensor.getDataPtr(), LazyTensorExecutionStatus::kINPUT);

    auto internal_tensor = hb_tensor.EvaluateTensorData(false);
    internal_tensor.unsafeGetTensorImpl()->set_wrapped_number(true);

    // Actual Copy is done during JIT graph creation/lowering
    context->copy_scalar_to_hpu_tensor_list.emplace_back(
        tensor, internal_tensor);
  };

  func();
  RUNNING_HASH_COMBINE_TENSOR(t);
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

  auto context = get_device_lazy_execution_context();
  static uint64_t hit_count, miss_count;
  auto dtype = options.dtype().toScalarType();

  std::lock_guard<std::mutex> lock(context->GetScalarToTensorMutex());
  auto map_it =
      context->scalar_to_tensor_map.find(std::make_pair(alpha, dtype));
  if (map_it == context->scalar_to_tensor_map.end()) {
    if (false == GET_ENV_FLAG_NEW(PT_HPU_SCALAR_H2D_COPY_MULTIPLE)) {
      alpha_tensor = at::tensor(alpha).to(dtype).to(c10::kHPU, true);
    } else {
      alpha_tensor = append_to_batch_h2d_list(at::tensor(alpha).to(dtype));
    }

    // Add to scalar value to device tensor cache
    context->scalar_to_tensor_map[std::make_pair(alpha, dtype)] = alpha_tensor;
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
  RUNNING_HASH_COMBINE_TENSOR(alpha_tensor);
  return alpha_tensor;
}

Tensor& copy_hpu_lazy_D2D(
    Tensor& self_,
    const Tensor& src_,
    bool non_blocking) {
  PT_LAZY_TRACE;

  auto is_src_const = habana::is_tensor_const(src_);
  auto const_tensor_id = habana::get_tensor_const_id(src_);
  habana::set_tensor_const(self_, is_src_const, const_tensor_id);

  auto src = HbLazyTensorViews::get_recent_base_tensor(src_);
  auto self = HbLazyTensorViews::get_recent_base_tensor(self_);

  habana::set_tensor_const(self, is_src_const, const_tensor_id);
  HbLazyTensor hb_tensor = GetHbLazyTensor(src);

  auto self_dtype = (self.scalar_type() == c10::ScalarType::Double)
      ? c10::ScalarType::Float
      : self.scalar_type();

  bool no_conversion = (src.scalar_type() == self.scalar_type());

  if (no_conversion && hb_tensor.IsExecutionInProgress()) {
    auto context = habana_lazy::get_device_lazy_execution_context();
    context->JoinPendingLaunchThread();
  }
  RUNNING_HASH_COMBINE_OPERATOR(hpu::copy_D2D, {self, src, no_conversion});
  auto op_func =
      [self_, src_, self_dtype, non_blocking, no_conversion]() mutable {
        ir::NodePtr node;
        std::vector<at::Tensor> input_pt_vec;
        auto is_src_const = habana::is_tensor_const(src_);
        auto src_const_id = habana::get_tensor_const_id(src_);
        habana::set_tensor_const(self_, is_src_const, src_const_id);
        // pick the most recent version of src tensor
        auto src = HbLazyTensorViews::get_recent_base_tensor(src_);
        auto self = HbLazyTensorViews::get_recent_base_tensor(self_);
        habana::set_tensor_const(self, is_src_const, src_const_id);
        HbLazyTensor hb_tensor = GetHbLazyTensor(src);
        auto hlresult = GetHbLazyTensor(self);

        auto layout_format = hb_tensor.GetTensorLayout();
        hlresult.SetTensorLayout(layout_format);

        if (no_conversion) {
          // If both src and dst are already processed ,  go and do the DMA dont
          // wait Else , If we already have storage in dst, add memcopy node to
          // lazy graph and we want to copy to existing tensor and not a new one
          // Kernel expects us to pass dst as second input in that case
          auto src_id = hb_tensor.getTensorUniqueId();
          auto dst_id = hlresult.getTensorUniqueId();
          if (src_id == dst_id) {
            return;
          }

          // graph cycle happens in squad 8x with view table mechanism
          // %id:3646 = hpu::as_strided_lazy(%id:18.1, %89, %90, %91)
          // %id:18 = hpu::habana_d2d_memcpy_other(%id:3646, %id:18.1)
          auto src_parent = HbLazyTensorViews::get_base_tensor(src);
          auto src_parent_id = GetHbLazyTensorId(src_parent);

          if (src_parent_id == dst_id) {
            return;
          }

          // Handle views and lhs slice
          auto is_view = HbLazyTensorViews::HandleViewsD2D(src, self);
          if (is_view == false) {
            AddMemcpy(src, self);
          }
        } else {
          node =
              std::make_shared<ir::Cast>(src, self.scalar_type(), non_blocking);

          auto& params_opt = GetHbLazyTensor(self).getDataPtr()->stride_params;
          if (params_opt.has_value()) {
            // add strided insert at the cast output
            at::TensorOptions options = src.options().dtype(self.scalar_type());
            auto src_cast = empty_hpu_lazy(
                src.sizes(), options, src.suggest_memory_format(), false);

            habana::set_tensor_const(src_cast, is_src_const, src_const_id);
            auto hl_src_cast = GetHbLazyTensor(src_cast);

            hl_src_cast.IrSetNode(node);
            flush_op(1);

            HbLazyTensorViews::HandleViewsD2D(src_cast, self);
          } else {
            LazyOp<at::Tensor> k{
                "hpu::cast", {src, self_dtype}, {src.sizes().vec()}};
            k.set_scalar_types({self_dtype});
            k.call(self);
          }
        }
      };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(copy_, op_func, self_);
}

at::Tensor handleWeightTensorLayout(const Tensor& src) {
  PT_LAZY_TRACE;
  habana_helpers::print_tensor_debug(src);
  /*  Synapse Layout nomenclature:
        rsck = {3, 2, 0, 1};
        qrsck = {4, 3, 0, 1, 2};
      PT layout struct:
        LayoutFormatDims = { N,C,H,W}
        LayoutFormatWithDepthDims = {N,C,D,H,W}
  */
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
  auto hb_tensor = GetHbLazyTensor(src);
  auto tensor_data = hb_tensor.EvaluateTensorData();

  return tensor_data;
}

void d2h_maybe_eval(const Tensor& src, bool async = false) {
  if (GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_DISABLED) {
    auto hl_t = GetHbLazyTensor(src);
    auto& params_opt = hl_t.getDataPtr()->stride_params;
    if (params_opt.has_value()) {
      hl_t = GetHbLazyTensor(
          HbLazyTensorViews::get_recent_base_tensor(params_opt.value().base));
    }

    if (hl_t.CurrentIrValue() && !hl_t.CurrentIrValue().IsHpuInputNode()) {
      PT_LAZY_DEBUG("Triggering mark_step before D2H copy");
      PT_IRGRAPH_DEBUG("step marker due to d2h_maybe_eval");
      HbLazyTensor::StepMarker({}, nullptr, {}, async);
    }
  }
}

void copy_hpu_lazy_D2H_internal(
    at::Tensor& self,
    at::Tensor& _src,
    bool non_blocking,
    synapse_helpers::hpuStream_t hpu_stream) {
  PT_LAZY_TRACE;

  auto hb_tensor = GetHbLazyTensor(_src);
  if (!hb_tensor.isStorageAttached()) {
    auto storage = _src.storage();
    if (storage.data_ptr() == nullptr) {
      c10 ::Allocator* allocator;
      allocator = habana::getHABANADeviceAllocator();
      int64_t nelements = multiply_integers(_src.sizes());
      int elem_size = _src.dtype().itemsize();
      int64_t size_bytes = nelements * elem_size;
      storage = c10::make_intrusive<StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size_bytes,
          allocator->allocate(nelements * elem_size),
          allocator,
          true);
    }
    auto at_internal_tensor = AtenInternalHbTensor(
        std::move(storage),
        _src.dtype(),
        DATA_TENSOR,
        _src.sizes(),
        _src.strides(),
        _src.options().memory_format_opt());
    hb_tensor.SetTensorData(at_internal_tensor);
  }

  hb_tensor.EvaluateTensorData();

  // If _src is a lazy tensor make sure the execution till the point of _src
  // getting flled has finished before we start copying
  auto tensor_data = handleWeightTensorLayout(_src);

  auto type = hb_tensor.getTensorOriginalType();
  // This path is disabled for now, when we return back from Habana to
  // CPU we can check if the original tensor was long/double , if soe we
  // can upscale it and send it back. For now we just send the 32bit
  // tensor that Habana holds

  if (type != typeMetaToScalarType(_src.dtype())) {
    // If we need to upscale the CPU tensor using the .to for now
    // It rebinds the self reference to the new tensor
    // We need to check the memory deletion of the original tensor created
    // by PT
    PT_LAZY_DEBUG(
        "WARNING: We are hitting a case in H2D where the PyTorch tensor original data types mismatch.");
    self = self.to(_src.dtype());
    self = copy_hpu_(self, tensor_data, non_blocking, hpu_stream);
    self = self.to(type);
  } else {
    auto tensor_data_ = tensor_data;
    if (_src.dtype() != tensor_data.dtype()) {
      auto tmp_hb_tensor = GetHbLazyTensor(_src);
      // try to find view_dtype in view chain (if available)
      while (tmp_hb_tensor.getDataPtr()->stride_params.has_value()) {
        const auto& stride_param =
            tmp_hb_tensor.getDataPtr()->stride_params.value();
        if (stride_param.optype == kStridedOpViewDtype) {
          tensor_data_ = at::empty_like(
              self, _src.options(), _src.suggest_memory_format());
          tensor_data_.unsafeGetTensorImpl()->set_storage_keep_dtype(
              tensor_data.storage());
          break;
        }
        tmp_hb_tensor = GetHbLazyTensor(stride_param.parent);
      }
    }
    self = copy_hpu_(self, tensor_data_, non_blocking, hpu_stream);
  }
  // No need to CreateHbLazyTensor for self as it is on CPU
  habana_lazy::PermuteTensors::handlePermutedTensor(_src, self, non_blocking);
}

void copy_hpu_lazy_D2H_async(
    at::Tensor& self,
    at::Tensor& _src,
    synapse_helpers::hpuStream_t hpu_stream,
    uint64_t launch_jobid) {
  PT_LAZY_TRACE;
  if (!self.defined()) {
    PT_LAZY_FATAL(
        "D2H src tensor expired in non_blocking scenario. Ensure the src cpu tensor is held until data is copied.");
    return;
  }
  auto context = habana_lazy::get_device_lazy_execution_context();
  context->m_async_d2h_context = true;
  context->m_launch_thread_context = true;
  copy_hpu_lazy_D2H_internal(self, _src, true, hpu_stream);
  context->m_async_d2h_context = false;
  context->m_launch_thread_context = false;
  context->DelFromJobidStreamidMap(launch_jobid);
}

Tensor& copy_hpu_lazy_D2H(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;

  habana_lazy::NoAccThread no_acc_thread;

  // This situation should not occur
  // Throwing an exception here for now to catch any cases that arise
  TORCH_CHECK(
      IsHbLazyTensor(src),
      "Habana Lazy : trying to copy back a tensor which does not have a lazy tensor");

  bool is_pinned = habana::PinnedMemoryAllocator_is_pinned(self.data_ptr());

  if (non_blocking && !is_pinned) {
    PT_LAZY_DEBUG(
        "WARNING: NonBlocking D2H async is supported only with pinned destination tensor.");
  }
  auto context = habana_lazy::get_device_lazy_execution_context();

  // if non_blocking then handle seperately as we dont want wait to finish
  // Launch thread execution.
  bool async_d2h_thread = false;
  // Take the async flow only if any launch thread is under execution..
  // otherwise there wont be any wait and we dont need async d2h thread
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM) &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_D2H_ASYNC_THREAD) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1) && non_blocking && is_pinned &&
      context->m_launch_thread_handle.valid()) {
    async_d2h_thread = true;
  } else {
    context->JoinPendingLaunchThread();
  }

  // Remove this SBS check, as of now prepare sbs inputs on calls .to operator
  // and on inplace ops that triggers this mark step, which leads to graph
  // evaluation and we end up losing input tensor.
  d2h_maybe_eval(src, (async_d2h_thread) ? true : false);
  // At this point markstep is executed. In non-blocking case, it will ensure do
  // launch thread join in the async thread.

  // handle views
  auto _src = HbLazyTensorViews::HandleViewsD2H(src);

  // suggest_memory_format() uses strides and sizes to determine the memory
  // format. Depending on strided_view's stride params, the self (and src)
  // memory format can be incorrecly mapped to ch last or ch last 3d. Refer:
  // LazyBasicKernelTest.noncontiguous. Use backend tensors memory format to
  // correctly identify the memory format
  self = self.contiguous(c10::MemoryFormat::Contiguous);

  auto stream = c10::hpu::getCurrentHPUStream();
  if (async_d2h_thread) {
    // Queue D2H to the execution Thread, so that the order of execution will be
    // preserved.
    // Creating new thread for this will have synchronization issues with
    // execution thread.
    size_t launch_jobid = context->GetUniqueJobId();
    context->AddToJobidStreamidMap(launch_jobid, stream.stream());

    context->m_launch_thread_handle =
        SingleTonExecThreadPool::getInstance().enqueue(
            copy_hpu_lazy_D2H_async, self, _src, stream, launch_jobid);
  } else {
    copy_hpu_lazy_D2H_internal(self, _src, non_blocking, stream);
  }

  habana_lazy::StageSubmission::getInstance().setStageSubmissionFlow(
      habana_lazy::StageSubmission::Mode::SET_WHEN_D2H_COPY);

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

// Use physical permute for internal bridge calls.
static Tensor permute_hpu_lazy_phy(const Tensor& self, IntArrayRef dims_in) {
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
        : LazyOp<T>("aten::permute", vector_of_inputs, nullptr, -1) {}

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
      habana::get_and_set_tensor_const(self, result);
      return result;
    }
  };

  Kernel kernel{vector_of_inputs};
  RUN_MAYBE_WITH_ACC_THREAD(permute, kernel);
}

Tensor& copy_hpu_lazy_H2D(Tensor& self, const Tensor& src_, bool non_blocking) {
  PT_LAZY_TRACE;

  habana_lazy::NoAccThread no_acc_thread;

  bool processed = false;
  auto is_src_const = habana::is_tensor_const(src_);
  auto const_tensor_id = habana::get_tensor_const_id(src_);
  habana::set_tensor_const(self, is_src_const, const_tensor_id);
  auto src = src_.contiguous(src_.suggest_memory_format());
  InitSizesAndStrides(
      self,
      c10::nullopt,
      self.sizes(),
      c10::nullopt,
      self.suggest_memory_format());
  auto exec_mode = get_habana_lazy_executor().getExecutionMode();
  if (exec_mode != kLOWERING) {
    auto self_hb_tensor = GetOrCreateHbLazyTensor(self, self.device());
    // WE need to add storage if it wasnt created
    // right now as soon as we do a H2D transfer, we create memory and mark
    // executed
    auto isStorageAttached = self_hb_tensor.isStorageAttached();
    if (!isStorageAttached) {
      auto storage = self.storage();
      if (storage.data_ptr() == nullptr) {
        c10 ::Allocator* allocator;
        allocator = habana::getHABANADeviceAllocator();
        int64_t nelements = multiply_integers(self.sizes());
        int elem_size = self.dtype().itemsize();
        int64_t size_bytes = nelements * elem_size;
        storage = c10::make_intrusive<StorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            size_bytes,
            allocator->allocate(nelements * elem_size),
            allocator,
            /*resizeable=*/true);
      }
      Tensor at_internal_tensor = AtenInternalHbTensor(
          std::move(storage),
          self.dtype(),
          c10::nullopt,
          src.sizes(),
          c10::nullopt,
          src.suggest_memory_format());
      // Setup the tensor sizes & strides for tensor with dim = 4, else for
      // now assuming contiguous
      habana::set_tensor_const(
          at_internal_tensor, is_src_const, const_tensor_id);
      self_hb_tensor.SetTensorData(at_internal_tensor);
      self_hb_tensor.SetIsConstTensor(is_src_const, const_tensor_id);
    }
  }
  auto new_tensor = preProcessIfLongorDouble(src, self, processed);

  // Get the internal tensor for copy kernel
  // First get the lazy tensor
  auto self_hb_tensor = GetOrCreateHbLazyTensor(self, self.device());
  auto context = get_device_lazy_execution_context();
  if (self_hb_tensor.IsExecutionInProgress()) {
    context->JoinPendingLaunchThread();
  }

  if (self_hb_tensor.CurrentIrValue() &&
      !self_hb_tensor.CurrentIrValue().IsHpuInputNode()) {
    PT_LAZY_DEBUG("Triggering mark_step before H2D copy");
    PT_IRGRAPH_DEBUG("step marker due to H2D copy");
    HbLazyTensor::StepMarker({});
  }

  // Set the tensor as input and mark as input
  self_hb_tensor.IrReconnectAsInputNode();

  context->MarkTensorStatus(
      self_hb_tensor.getDataPtr(), LazyTensorExecutionStatus::kINPUT);
  auto& params_opt = self_hb_tensor.getDataPtr()->stride_params;
  // We need to mark this tensor as executed
  // As this will be an input coming from host side, its doesnt need further
  // execution and is ready for consumption as input
  // This is the internal tensor, it isn't a lazy tensor
  auto self_internal_tesor{self_hb_tensor.EvaluateTensorData()};

  HABANA_ASSERT(!TryGetHbLazyTensor(self_internal_tesor));

  auto smeta{get_storage_extra_meta(self_internal_tesor)};
  if (smeta) {
    auto synapse_permute = smeta->get_memory_permutation();
    if (synapse_permute.size() != 0) {
      PT_LAYOUTS_DEBUG(
          "clearing memory permute, id ",
          self_hb_tensor.getTensorUniqueId(),
          " permute ",
          VecToString(synapse_permute))
      smeta->set_memory_permutation({});
    }
  }
  habana::set_tensor_const(self_internal_tesor, is_src_const, const_tensor_id);

  // self may have been resized, so re-set its size and strides
  self_internal_tesor.unsafeGetTensorImpl()->set_sizes_and_strides(
      self.sizes(), self.strides());
  if (processed) {
    auto internal_tensor_from_copy = copy_hpu_(
        self_internal_tesor,
        new_tensor,
        non_blocking,
        c10::hpu::getCurrentHPUStream());
    // We should get back the same internal tensor passed to copy
    HABANA_ASSERT(
        self_internal_tesor.storage().data_ptr() ==
        internal_tensor_from_copy.storage().data_ptr());
  } else {
    auto internal_tensor_from_copy = copy_hpu_(
        self_internal_tesor,
        src,
        non_blocking,
        c10::hpu::getCurrentHPUStream());
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
    self = permute_hpu_lazy_phy(self, permute_dims);
  }

  // TODO : Handle view table update of channels_last tensor
  if (params_opt.has_value()) {
    strided_insert_hpu_lazy(self, self, false);
  }

  habana::set_tensor_const(self, is_src_const, const_tensor_id);

  // Return the self tensor, as copy_hpu_ doesn't create a new tensor and
  // returns the dst
  flush_op(1);
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

  auto is_src_const = habana::is_tensor_const(src);
  auto const_tensor_id = habana::get_tensor_const_id(src);
  habana::set_tensor_const(self, is_src_const, const_tensor_id);

  // If it isnt a device to device copy, we are transferring data to and
  // from CPU. This becomes an execution step point and we need to flush
  // graph execution NOW to generate tensor data where required as we are in
  // lazy mode. Otherwise we have to add the copy induced nodes(like cast)
  // to lazy graph for execution later
  if (!is_d2d_copy) {
    if (src_device == c10::DeviceType::CPU) {
      if (IsHbLazyTensor(self)) {
        auto& self_params_opt =
            GetHbLazyTensor(self).getDataPtr()->stride_params;
        if (self_params_opt.has_value()) {
          // self = base.view()
          // self.copy_(cpu_tensor)
          // lower it as insert_t = cpu_tensor.to('hpu')
          // strided_insert/slice_insert(self, insert_t, view_params)
          auto insert_t = empty_hpu_lazy(
              src.sizes(),
              self.options(),
              c10::MemoryFormat::Contiguous,
              false);
          habana::set_tensor_const(insert_t, is_src_const, const_tensor_id);
          copy_hpu_lazy_H2D(insert_t, src, non_blocking);
          HbLazyTensorViews::HandleViewsD2D(insert_t, self);
          habana::set_tensor_const(self, is_src_const, const_tensor_id);
          return self;
        }
      }
      self = copy_hpu_lazy_H2D(self, src, non_blocking);
    } else if (src_device == c10::DeviceType::HPU) {
      self = copy_hpu_lazy_D2H(self, src, non_blocking);
    }
  } else {
    self = copy_hpu_lazy_D2D(self, src, non_blocking);
  }

  habana::set_tensor_const(self, is_src_const, const_tensor_id);
  return self;
}

std::vector<uint64_t> get_strided_view_stride_data(
    at::IntArrayRef& stride,
    int64_t& offset) {
  std::vector<uint64_t> stride_data_vec;

  stride_data_vec.push_back(static_cast<uint64_t>(stride.size()));
  stride_data_vec.push_back(static_cast<uint64_t>(offset));
  for (auto it = stride.rbegin(); it != stride.rend(); ++it) {
    stride_data_vec.push_back(static_cast<uint64_t>(*it));
  }
  size_t fill_dim = (SYN_MAX_TENSOR_DIM + 1) - stride.size();
  for (size_t i = 0; i < fill_dim; i++) {
    stride_data_vec.push_back(static_cast<uint64_t>(0));
  }

  return stride_data_vec;
}

bool IsStridedViewRatioUndefined(
    std::vector<int64_t>& self_strides,
    std::vector<int64_t>& stride_sizes) {
  if (self_strides.size() != stride_sizes.size()) {
    return true;
  }
  auto len = self_strides.size();
  for (uint64_t i = 0; i < len; i++) {
    if (stride_sizes[i] < self_strides[i]) {
      return true;
    }
  }
  return false;
}

ir::NodePtr strided_view_h2d(
    const Tensor& self,
    Tensor& out_size_st,
    Tensor& offset_st,
    at::IntArrayRef& orig_stride,
    at::IntArrayRef& stride,
    int64_t& offset,
    std::string& node_str) {
  ir::NodePtr node = nullptr;
  std::vector<uint64_t> stride_data_vec =
      get_strided_view_stride_data(stride, offset);
  auto self_strides = orig_stride.vec();
  auto stride_sizes = stride.vec();

  auto stride_st = empty_hpu_lazy(
      stride_data_vec.size() * 2,
      self.options(),
      self.suggest_memory_format(),
      false,
      HOST_TO_DEVICE_TENSOR);
  auto hl_stride_st = GetOrCreateHbLazyTensor(stride_st, c10::kHPU);
  auto hl_stride_internal = hl_stride_st.CurrentTensorAttached().value();
  auto tmeta{get_tensor_extra_meta(hl_stride_internal)};
  tmeta->set_host_data(
      stride_data_vec.data(),
      stride_data_vec.size(),
      sizeof(uint64_t),
      HostDataType::UINT64_T);

  if (IsStridedViewRatioUndefined(self_strides, stride_sizes)) {
    if (node_str == "hpu::strided_view_out_ds") {
      node_str = "hpu::strided_view_out_orig_ds_h2d";
    } else {
      node_str = "hpu::strided_view_orig_ds_h2d";
    }

    tmeta->set_H2D_data_for_bucketing();
    node = std::make_shared<ir::StridedView>(
        self, out_size_st, stride_st, node_str);
  } else {
    if (node_str == "hpu::strided_view_out_ds") {
      node_str = "hpu::strided_view_out_ds_h2d";
    } else {
      node_str = "hpu::strided_view_ds_h2d";
    }

    auto offset_t = GetHbLazyTensor(offset_st);
    auto tensor_offset = offset_t.CurrentTensorAttached().value();

    auto tmeta_offset{get_tensor_extra_meta(tensor_offset)};

    // Mark this front end shape tensor as it does not need synapse tensor.
    // It carries stride_ratios info for BE lowering kernel.
    tmeta_offset->set_H2D_frontend_shape_tensor();

    std::vector<int64_t> stride_ratios;
    auto self_strides = orig_stride.vec();
    auto stride_sizes = stride.vec();
    auto len = stride_sizes.size();
    for (uint64_t i = 0; i < len; i++) {
      stride_ratios.push_back(stride_sizes[i] / self_strides[i]);
    }
    tmeta_offset->get_shape_struct().set_strides_tensor_shape(stride_sizes);
    tmeta_offset->get_shape_struct().set_stride_ratio(stride_ratios);
    PT_DYNAMIC_SHAPE_DEBUG(
        "Setting stride ratio = ", stride_ratios, " offset = ", offset);

    node = std::make_shared<ir::StridedView>(
        self, out_size_st, stride_st, offset_st, node_str);
  }

  return node;
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

  auto tmeta_self{get_tensor_extra_meta(self)};
  auto tmeta_internal_tensor{get_tensor_extra_meta(at_internal_tensor)};
  auto layout_format = tmeta_self->get_tensor_layout();
  tmeta_internal_tensor->set_tensor_layout(layout_format);
  tmeta_internal_tensor->set_is_const_tensor(tmeta_self->is_const_tensor());
  tmeta_internal_tensor->set_const_id(tmeta_self->get_const_id());

  return at_internal_tensor;
}

ir::NodePtr create_as_strided_node(
    const Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset,
    bool is_out) {
  return create_as_strided_node(
      self, size, stride, self.sizes(), self.strides(), storage_offset, is_out);
}

ir::NodePtr create_as_strided_node(
    const Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    at::IntArrayRef orig_size,
    at::IntArrayRef orig_stride,
    c10::optional<int64_t> storage_offset,
    bool is_out) {
  ir::NodePtr node = nullptr;
  auto self_strides = orig_stride.vec();
  auto stride_sizes = stride.vec();
  auto offset = storage_offset.value_or(self.storage_offset());
  if (habana_helpers::GetRefineDynamicShapeStatus()) {
    std::string node_str =
        (is_out) ? "hpu::strided_view_out_ds" : "hpu::strided_view_ds";

    PT_DYNAMIC_SHAPE_DEBUG(
        "Strided view Real size = ",
        orig_size.vec(),
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

    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED)) {
      return strided_view_h2d(
          self, out_size_st, offset_st, orig_stride, stride, offset, node_str);
    }

    if (IsStridedViewRatioUndefined(self_strides, stride_sizes)) {
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
      auto tmeta{get_tensor_extra_meta(tensor_size_st)};

      std::vector<int64_t> stride_ratios;
      auto self_strides = orig_stride.vec();
      auto stride_sizes = stride.vec();
      auto len = stride_sizes.size();
      for (uint64_t i = 0; i < len; i++) {
        stride_ratios.push_back(stride_sizes[i] / self_strides[i]);
      }
      tmeta->get_shape_struct().set_strides_tensor_shape(stride_sizes);
      tmeta->get_shape_struct().set_stride_ratio(stride_ratios);
      PT_DYNAMIC_SHAPE_DEBUG(
          "Setting stride ratio = ", stride_ratios, " offset = ", offset);

      node = std::make_shared<ir::StridedView>(
          self, out_size_st, offset_st, node_str);
    }
  } else {
    std::string node_str =
        (is_out) ? "hpu::strided_view_out" : "hpu::strided_view";

    node =
        std::make_shared<ir::StridedView>(self, size, stride, offset, node_str);
  }
  return node;
}

// THis kernel has two paths, lowering and lazy
// During lazy we set up the as strided tensor meta data
// when we get a call back from lowering, we attache the tensor from same memory
// as source
Tensor as_strided_hpu(
    const Tensor& self,
    SymIntArrayRef size,
    SymIntArrayRef stride,
    c10::optional<SymInt> offset) {
  PT_LAZY_TRACE;
  auto storage_offset_val =
      offset.has_value() ? offset.value().expect_int() : self.storage_offset();

  auto size_in = C10_AS_INTARRAYREF_SLOW(size);
  auto stride_in = C10_AS_INTARRAYREF_SLOW(stride);
  // lazy within lazy. as strided node is not here. Only the view table update
  // happens here

  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) == true)) {
    auto out = HbLazyTensorViews::process_strided_view(
        self, size_in, stride_in, storage_offset_val, true);
    return out;
  } else {
    auto out = HbLazyTensorViews::add_strided_view_node(
        self,
        size_in,
        stride_in,
        storage_offset_val,
        true /*is_update_view*/,
        c10::nullopt);
    if (get_habana_lazy_executor().getExecutionMode() != kLOWERING) {
      flush_op(1);
    }
    return out;
  }
}; // namespace habana_lazy

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

  // lazy within lazy. as strided node is not here. Only the view table update
  // happens here
  auto storage_offset_val = storage_offset.value_or(self.storage_offset());

  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) == true)) {
    auto out = HbLazyTensorViews::process_strided_view(
        self, size_in, stride_in, storage_offset_val, true);
    return out;
  } else {
    auto out = HbLazyTensorViews::add_strided_view_node(
        self,
        size_in,
        stride_in,
        storage_offset_val,
        true /*is_update_view*/,
        c10::nullopt);

    habana::get_and_set_tensor_const(self, out);
    if (get_habana_lazy_executor().getExecutionMode() != kLOWERING) {
      flush_op(1);
    }
    return out;
  }
}

void as_strided_hpu_lazy_inplace_parralel_impl(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    IntArrayRef orig_size,
    IntArrayRef orig_stride,
    c10::optional<int64_t> storage_offset) {
  // We only support contiguous chunks of data to be taken as strided,
  // as Device doesnt support strided tensors we dont support that case
  ir::NodePtr node = create_as_strided_node(
      self, size, stride, orig_size, orig_stride, storage_offset);

  if (node != nullptr) {
    auto hb_result = GetHbLazyTensor(self);
    // update of lazy tensor size is required for permute pass to see output
    // with updated shape
    hb_result.setTensorSize(size);
    hb_result.IrSetNode(node);

    auto context = get_device_lazy_execution_context();
    context->MarkTensorStatus(
        hb_result.getDataPtr(), LazyTensorExecutionStatus::kREGISTERED);
    flush_op(1);
  } else {
    TORCH_CHECK(
        0,
        "as_strided_ called with strides creating non-contiguous output tensor not supported");
  }
};
const Tensor& as_strided_hpu_lazy_(
    const Tensor& self,
    SymIntArrayRef _size,
    SymIntArrayRef _stride,
    c10::optional<SymInt> _storage_offset) {
  auto size = C10_AS_INTARRAYREF_SLOW(_size);
  auto stride = C10_AS_INTARRAYREF_SLOW(_stride);
  auto orig_size = self.sizes().vec();
  auto orig_stride = self.strides().vec();
  auto storage_offset = _storage_offset.has_value()
      ? _storage_offset.value().expect_int()
      : self.storage_offset();
  self.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  handle_collective(self);
  auto func = [self,
               size = size.vec(),
               stride = stride.vec(),
               orig_size = std::move(orig_size),
               orig_stride = std::move(orig_stride),
               storage_offset = std::move(storage_offset)]() {
    as_strided_hpu_lazy_inplace_parralel_impl(
        self, size, stride, orig_size, orig_stride, storage_offset);
  };
  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(as_strided_, func, self);
}

at::Tensor& set_source_Storage_storage_offset(
    at::Tensor& self,
    at::Storage source,
    at::SymInt storage_offset_,
    at::SymIntArrayRef size_,
    at::SymIntArrayRef /*stride*/) {
  PT_LAZY_TRACE
  auto storage_offset = storage_offset_.expect_int();
  auto size = C10_AS_INTARRAYREF_SLOW(size_);
  // TODO Handle stride

  auto lazy_ten = GetHbLazyTensor(self);
  auto impl = lazy_ten.getAttachedTensorImpl();
  HABANA_ASSERT(impl, "impl is invalid");
  impl->set_storage_keep_dtype(std::move(source));
  impl->set_storage_offset(storage_offset);
  self.resize_(size, self.suggest_memory_format());

  return self;
}

void view_hpu_lazy_parallel_impl(
    const Tensor& self_,
    IntArrayRef size,
    const Tensor& out) {
  Tensor self = self_;
  auto& self_params_opt = GetHbLazyTensor(self).getDataPtr()->stride_params;
  // multilevel view optimization
  // v1 = view(a, out_size1)
  // v2 = view(v1, out_size2)
  // The above sequence can be compressed to v2 = view(a, out_size2)
  if (self_params_opt.has_value()) {
    auto& self_params = self_params_opt.value();
    if (self_params.optype == kStridedOpView) {
      self = self_params.parent;
      PT_VIEWTABLE_DEBUG("invoked multilevel view optimization");
    }
  }
  lazy_view_fallback_handle(
      self, out, [size = size.vec()](const Tensor& self, StrideParams& params) {
        // There could be some cases where slice/select/etc followed by view,
        // in those cases use as_strided instead of using the ViewOP.
        params.optype = kStridedOpView;

        PT_VIEWTABLE_DEBUG(
            "view fallback- tensor id: ",
            GetHbLazyTensorId(self),
            "size ",
            size);
      });
}

Tensor view_hpu(const Tensor& self_, SymIntArrayRef size) {
  PT_LAZY_TRACE;
  auto size_ = C10_AS_INTARRAYREF_SLOW(size);
  auto inferred_size = habana_helpers::infer_size(size_, self_.numel());
  auto stride =
      at::detail::computeStride(self_.sizes(), self_.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto stride_value = *stride;

  handle_collective(self_);
  auto out = as_strided_hpu_lazy(
      self_, inferred_size, stride_value, self_.storage_offset());

  // no need to create view table
  if (lazyEagerOptimizedViewHandling()) {
    return out;
  }

  auto func = std::bind(view_hpu_lazy_parallel_impl, self_, size_.vec(), out);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_VIEW_OPS_MODE) != 0) {
    RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(view, func, out);
  } else {
    func();
    return out;
  }
}

// Computes the strides for view_dtype output when the view dtype is
// smaller than the original dtype
inline DimVector compute_strides_for_view_dtype_downsize(
    IntArrayRef old_strides,
    int64_t size_ratio,
    ScalarType old_dtype,
    ScalarType new_dtype) {
  const int64_t ndim = old_strides.size();

  TORCH_CHECK(
      old_strides[ndim - 1] == 1,
      "self.stride(-1) must be 1 to view ",
      old_dtype,
      " as ",
      new_dtype,
      " (different element sizes), but got ",
      old_strides[ndim - 1]);

  DimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    new_strides[dim_idx] = old_strides[dim_idx] * size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

// Computes the strides for view_dtype output when the view dtype is
// larger than the original dtype
inline DimVector compute_strides_for_view_dtype_upsize(
    IntArrayRef old_strides,
    int64_t size_ratio,
    ScalarType old_dtype,
    ScalarType new_dtype) {
  const int64_t ndim = old_strides.size();
  TORCH_CHECK(
      old_strides[ndim - 1] == 1,
      "self.stride(-1) must be 1 to view ",
      old_dtype,
      " as ",
      new_dtype,
      " (different element sizes), but got ",
      old_strides[ndim - 1]);

  DimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    TORCH_CHECK(
        (old_strides[dim_idx] % size_ratio) == 0,
        "self.stride(",
        dim_idx,
        ") must be divisible by ",
        size_ratio,
        " to view ",
        old_dtype,
        " as ",
        new_dtype,
        " (different element sizes), ",
        "but got ",
        old_strides[dim_idx]);

    new_strides[dim_idx] = old_strides[dim_idx] / size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

Tensor view_dtype_hpu(const Tensor& self, ScalarType dtype) {
  PT_LAZY_TRACE;
  if (self.scalar_type() == dtype) {
    return self;
  }
  const auto type_meta = c10::scalarTypeToTypeMeta(dtype);
  TORCH_CHECK(
      !self.is_conj(),
      "torch.Tensor.view is not supported for conjugate view tensors when converting to a different dtype.");
  TORCH_CHECK(
      !self.is_neg(),
      "torch.Tensor.view is not supported for tensors with negative bit set when converting to a different dtype.");

  int64_t self_element_size = self.element_size();
  int64_t new_element_size = static_cast<int64_t>(type_meta.itemsize());

  // Handle bool dtype when self_element_size == new_element_size
  if (self_element_size == new_element_size && dtype == c10::ScalarType::Bool) {
    return self.to(dtype);
  }

  auto new_tensor = view_hpu(self, fromIntArrayRefUnchecked(self.sizes()));
  auto* impl = new_tensor.unsafeGetTensorImpl();
  impl->set_storage_and_dtype(self.storage(), type_meta);
  if (self_element_size == new_element_size) {
    impl->set_storage_offset(self.storage_offset());
    impl->set_sizes_and_strides(self.sizes(), self.strides());
  } else if (self.dim() == 0) {
    TORCH_CHECK(
        false,
        "self.dim() cannot be 0 to view ",
        self.scalar_type(),
        " as ",
        dtype,
        " (different element sizes)");

  } else if (self_element_size > new_element_size) {
    // Downsizing element size

    int64_t size_ratio = self_element_size / new_element_size;
    auto new_strides = compute_strides_for_view_dtype_downsize(
        self.strides(), size_ratio, self.scalar_type(), dtype);

    auto old_sizes = self.sizes();
    DimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes[self.dim() - 1] *= size_ratio;

    auto new_storage_offset = size_ratio * self.storage_offset();

    impl->set_storage_offset(new_storage_offset);
    impl->set_sizes_and_strides(new_sizes, new_strides);

  } else {
    // Upsizing element size

    int64_t size_ratio = new_element_size / self_element_size;

    TORCH_CHECK(
        (self.size(-1) % size_ratio) == 0,
        "self.size(-1) must be divisible by ",
        size_ratio,
        " to view ",
        self.scalar_type(),
        " as ",
        dtype,
        " (different element sizes), ",
        "but got ",
        self.size(-1));

    TORCH_CHECK(
        (self.storage_offset() % size_ratio) == 0,
        "self.storage_offset() must be divisible by ",
        size_ratio,
        " to view ",
        self.scalar_type(),
        " as ",
        dtype,
        " (different element sizes), but got ",
        self.storage_offset());

    auto new_strides = compute_strides_for_view_dtype_upsize(
        self.strides(), size_ratio, self.scalar_type(), dtype);

    auto old_sizes = self.sizes();
    DimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes[self.dim() - 1] /= size_ratio;

    auto new_storage_offset = self.storage_offset() / size_ratio;

    impl->set_storage_offset(new_storage_offset);
    impl->set_sizes_and_strides(new_sizes, new_strides);
  }

  auto hb_tensor = GetHbLazyTensor(new_tensor);
  hb_tensor.setTensorOriginalType(dtype);
  auto& params_opt = hb_tensor.getDataPtr()->stride_params;
  TORCH_CHECK(params_opt.has_value(), "view_dtype: incorrect stride params");
  params_opt.value().optype = kStridedOpViewDtype;

  return new_tensor;
}

void add_tensor_hpu_lazy_parallel_impl(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  PT_LAZY_TRACE;

  auto alpha_double = alpha.toDouble();
  if (alpha_double != 1.0) {
    at::Tensor mul_out;
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
      // For lazy eager Skip scalar handling at FE
      mul_out = torch::mul(other, alpha);
    } else {
      at::Tensor alpha_tensor =
          get_tensor_for_scalar(alpha_double, other.options());

      auto hl_alpha = GetOrCreateHbLazyTensor(alpha_tensor, c10::kHPU);
      mul_out = torch::mul(other, alpha_tensor);
    }
    if (other.unsafeGetTensorImpl()->is_wrapped_number()) {
      // The operation has been split into intermediate multiply and then
      // again add op tensor produced by this split resulted in inappropriate
      // type deduction of whole add operation. alpha is always scalar, when
      // also other is scalar then marking intermediate as wrapped number is
      // also necessary to further proper deduction
      mul_out.unsafeGetTensorImpl()->set_wrapped_number(true);
    }
    add_tensor_hpu_lazy_parallel_impl(self, mul_out, 1.0, out);
  } else {
    LazyBinaryOp<at::Tensor> k{
        "hpu::add",
        {self, other, alpha},
        false,
        true,
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
      "hpu::add",
      {self, other, alpha},
      false,
      true,
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
  LazyOp<at::Tensor> op{"hpu::add", {self, other, alpha}, {self.sizes().vec()}};
  RUN_MAYBE_WITH_ACC_THREAD(add, op)
}

Tensor& add_scalar_hpu_lazy_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;

  // Handle scalar handling for lazy mode only at FE
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2) {
    auto other_tensor = get_tensor_for_scalar(other.toDouble(), self.options());
    return add_tensor_hpu_lazy_(self, other_tensor, alpha);
  }

  LazyOp<Tensor&> op("hpu::add_", {self, other, alpha});
  return op.call(self);
}

void add_tensor_hpu_lazy_inplace_parallel_impl(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  auto alpha_double = alpha.toDouble();
  if (alpha_double != 1.0) {
    at::Tensor mul_out;
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
      // For lazy eager Skip scalar handling at FE
      mul_out = torch::mul(other, alpha);
    } else {
      at::Tensor alpha_tensor =
          get_tensor_for_scalar(alpha_double, other.options());

      auto hl_alpha = GetOrCreateHbLazyTensor(alpha_tensor, c10::kHPU);
      mul_out = torch::mul(other, alpha_tensor);
    }
    add_tensor_hpu_lazy_inplace_parallel_impl(self, mul_out, 1.0);
  } else {
    LazyBinaryOp<Tensor&> op("hpu::add_", {self, other, alpha}, false, true);
    op.call(self);
  }
}

Tensor& add_tensor_hpu_lazy_(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;

  if (habana_lazy::AccThread::IsAccThreadEnabled()) {
    // try to construct DTypeHelper to make sure,
    // that type promotion does not throw due to incompatible dtypes
    at::IValue ivalue(self);
    auto output = c10::make_optional<const at::IValue*>(&ivalue);
    auto dtype_helper =
        habana_helpers::DTypeHelper::binary_op_with_type_promotion(
            {self, other}, output, true);
  }

  handle_collective(self);
  handle_collective(other);
  auto op_func = [self, other, alpha]() mutable {
    add_tensor_hpu_lazy_inplace_parallel_impl(self, other, alpha);
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(add_, op_func, self);
}

at::Tensor cast_to_32(const at::Tensor& self) {
  if (self.scalar_type() == c10::ScalarType::Long ||
      self.scalar_type() == c10::ScalarType::Byte ||
      self.scalar_type() == c10::ScalarType::Char ||
      self.scalar_type() == c10::ScalarType::Short) {
    return self.to(c10::ScalarType::Int);
  } else if (
      self.scalar_type() == c10::ScalarType::Double ||
      self.scalar_type() == c10::ScalarType::Half ||
      self.scalar_type() == c10::ScalarType::BFloat16) {
    return self.to(c10::ScalarType::Float);
  }
  return self;
}

c10::optional<at::Tensor> cast_weights(
    const c10::optional<at::Tensor>& weights) {
  if (weights.has_value() &&
      (weights.value().dtype() != c10::ScalarType::Int &&
       weights.value().dtype() != c10::ScalarType::Float)) {
    return c10::make_optional<at::Tensor>(cast_to_32(weights.value()));
  }
  return weights;
}

c10::ScalarType bincount_output_dtype(
    const c10::optional<at::Tensor>& weights) {
  if (!weights.has_value()) {
    return c10::ScalarType::Long;
  }
  return (weights.value().scalar_type() == c10::ScalarType::Float)
      ? c10::ScalarType::Float
      : c10::ScalarType::Double;
}

Tensor bincount_hpu_lazy(
    const Tensor& self,
    const c10::optional<Tensor>& weights,
    int64_t minlength) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  auto elements = self.numel();

  if (elements == 0) {
    auto shape = DimVector{minlength};
    return at::zeros(shape, TensorOptions(kHPU).dtype(at::kLong));
  }

  auto max_in_input = static_cast<int64_t>(at::max(self).item<int64_t>());
  int64_t length = std::max(max_in_input + 1, minlength);
  std::vector<int64_t> shape{length};
  // Add bincount node
  LazyOp<at::Tensor> hpu_op{
      "hpu::bincount_backend",
      {cast_to_32(self), length, cast_weights(weights)},
      {shape},
      0};
  hpu_op.SetOutputMetaFn(BinCountMeta);
  auto result_bincount = hpu_op.call();
  auto out_dtype = bincount_output_dtype(weights);
  if (result_bincount.scalar_type() != out_dtype) {
    return result_bincount.to(out_dtype);
  }
  return result_bincount;
}

at::Tensor prepare_hpu_tensor(const at::Tensor& t) {
  if (t.defined() && t.device().type() != c10::DeviceType::HPU) {
    // If the CPU tensor is a wrapped number, then use
    // get_tensor_for_scalar method to retrieve cached HPU tensors for
    // the scalar value
    if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
      // is_wrapped_number: True if a tensor was auto-wrapped from a
      // C++ or Python number.
      auto dtype = t.scalar_type();
      return get_tensor_for_scalar(
          t.item().toDouble(), at::TensorOptions().dtype(dtype));
    } else {
      // Use non_blocking .to()
      return t.to(c10::kHPU, true);
    }
  }
  return t;
}

Tensor& mul_out_hpu_lazy(
    const Tensor& self,
    const Tensor& raw_other,
    Tensor& out) {
  PT_LAZY_TRACE;
  // 8x all reduce optimization to avoid out variant that requires tensor with
  // storage. //TODO enhance lazy op framework to convert out variant to out
  // of place variant
  auto other{prepare_hpu_tensor(raw_other)};
  RUNNING_HASH_COMBINE_OPERATOR(hpu::mul_out, {self, raw_other, out});
  auto shape_changed = self.sizes() != out.sizes();
  if (shape_changed) {
    out.unsafeGetTensorImpl()->set_sizes_contiguous(self.sizes());
  }

  handle_collective(self);
  handle_collective(other);
  handle_collective(out);

  auto func = [self, other, out, shape_changed]() mutable {
    auto& params_opt = GetHbLazyTensor(out).getDataPtr()->stride_params;

    if (params_opt.has_value()) {
      auto orig_out = out;
      auto temp = torch::mul(self, other);
      Tensor temp_cast = temp;
      if (temp.scalar_type() != orig_out.scalar_type()) {
        // Cast temp tensor to orig_out tensor data type
        LazyOp<Tensor> k_{
            "hpu::cast", {temp, orig_out.scalar_type()}, {temp.sizes().vec()}};
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
      if (shape_changed) {
        hpu_op.set_shape_changed();
      }

      hpu_op.call(out);
    }
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(mul_out, func, out)
}

Tensor complex_hpu(const at::Tensor& real, const at::Tensor& imag) {
  return at::native::complex(real.to("cpu"), imag.to("cpu"));
}

Tensor constant_pad_hpu_lazy(
    const Tensor& self,
    at::SymIntArrayRef pad_sym,
    const Scalar& value) {
  PT_LAZY_TRACE;

  std::vector<int64_t> pad_int;
  for (auto& val : pad_sym) {
    pad_int.push_back(val.expect_int());
  }
  IntArrayRef pad = makeArrayRef(pad_int);

  auto sizes = PadOperator::compute_output_shape(self, pad);
  auto out = empty_hpu_lazy(
      sizes, self.options(), self.suggest_memory_format(), false);

  std::vector<int64_t> pad_vec = pad.vec();
  auto func = [pad_vec = std::move(pad_vec), out, self, value]() mutable {
    IntArrayRef pad = pad_vec;
    std::vector<at::IValue> vector_of_inputs;
    std::string op_name;
    if (habana_helpers::GetRefineDynamicShapeStatus()) {
      op_name = "hpu::constant_pad_nd_ht";
      std::vector<uint32_t> pad_ht_vec(MAX_DIMENSIONS_NUM * 2, 0);
      // assuming that "pad" has a pair of pad values corresponding to each
      // dim that needs to be padded.
      for (unsigned int i = 0; i < pad.size() / 2; i++) {
        // Host tensor layout 1D - 10 elements:
        // pad_before[0]...pad_before[4], pad_after[0] ... pad_after[4] (for
        // dimensionality IFM less then 5 some elements not in use)
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
      auto tmeta_shape_tensor_internal{
          get_tensor_extra_meta(hl_output_shape_tensor_internal, true)};
      if (tmeta_shape_tensor_internal) {
        tmeta_shape_tensor_internal->set_H2D_frontend_shape_tensor();
      }
      auto hl_params_shape = GetOrCreateHbLazyTensor(pad_tensor, c10::kHPU);

      auto hl_param_internal = hl_params_shape.CurrentTensorAttached().value();
      auto tmeta{get_tensor_extra_meta(hl_param_internal)};
      tmeta->set_host_data(
          pad_ht_vec.data(),
          pad_ht_vec.size(),
          sizeof(uint32_t),
          HostDataType::UINT32_T);
      vector_of_inputs = {self, pad_tensor, output_shape_tensor, value};
    } else {
      op_name = "hpu::constant_pad_nd_lazy";
      vector_of_inputs = {self, pad, value};
    }
    LazyOp<at::Tensor> k{
        op_name,
        vector_of_inputs,
        {PadOperator::compute_output_shape(self, pad)}};
    k.call(out);
  };
  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(constant_pad_nd, func, out)
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

// scatter_add is producing wrong value randomly
// https://jira.habana-labs.com/browse/SW-44742
Tensor& scatter_add_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_LAZY_TRACE;

  handle_collective(self);
  auto func = [self, dim_, index, src]() mutable {
    auto node =
        std::make_shared<habana_lazy::ir::ScatterAdd>(self, dim_, index, src);
    LazyOp<at::Tensor, ir::ScatterAdd> k{node, {self, dim_, index, src}};
    auto result = k.call();
    auto hl_self = GetOrCreateHbLazyTensor(self);
    // Create MemCopy operator to copy value into self
    AddMemcpy(result, self);
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(scatter_add_, func, self)
}

static bool check_for_advanced_indexing(
    const c10::List<c10::optional<at::Tensor>>& indices) {
  bool advanced_indexing = false;
  c10::ScalarType prev_scalar_type = c10::ScalarType::Long;
  bool first_scalar = true;

  if (indices.size() <= MAX_DIMS_FOR_ADVANCED_INDEXING) {
    for (c10::optional<at::Tensor> input_ind : indices) {
      auto input = input_ind.value_or(Tensor());
      if (!input.defined()) {
        advanced_indexing = true;
        break;
      } else {
        // if we are indexing using a mixture of long and boolean indices,then
        // also we will work in advanced indexing mode
        auto cur_scalar_type = input.scalar_type();
        if (first_scalar) {
          first_scalar = false;
        } else if (prev_scalar_type != cur_scalar_type) {
          advanced_indexing = true;
          break;
        }
        prev_scalar_type = cur_scalar_type;
      }
    }
  }
  return advanced_indexing;
}

static c10::List<c10::optional<at::Tensor>> check_for_boolean_advanced_indexing(
    const c10::List<c10::optional<at::Tensor>>& indices) {
  std::vector<c10::optional<at::Tensor>> bool_indices_vec;
  at::Tensor t_nz;
  bool has_bool_mask = false;
  for (c10::optional<at::Tensor> input_ind : indices) {
    auto input_temp = input_ind.value_or(Tensor());
    Tensor input;
    if (input.defined() &&
        (input_temp.device().type() != c10::DeviceType::HPU)) {
      input = input_temp.to(c10::kHPU);
    } else {
      input = input_temp;
    }
    if (!input.defined()) {
      bool_indices_vec.emplace_back(input);
    } else {
      auto cur_scalar_type = input.scalar_type();
      if (cur_scalar_type == c10::ScalarType::Bool) {
        has_bool_mask = true;
        auto nonzero_indices = habana_lazy::nonzero_hpu_lazy(input);
        t_nz = habana_lazy::squeeze_hpu_lazy(nonzero_indices, 1);
        if (t_nz.dim() > 1) {
          std::vector<int64_t> dims_sz_vec(t_nz.sizes()[1], 1);
          c10::IntArrayRef dims_sz(dims_sz_vec);
          auto nz_indices =
              habana_lazy::split_with_sizes_hpu_lazy(t_nz, dims_sz, 1);
          for (auto i : c10::irange((int)nz_indices.size())) {
            auto nzi = habana_lazy::squeeze_hpu_lazy(nz_indices.at(i), 1);
            bool_indices_vec.emplace_back(nzi);
          }
        } else {
          bool_indices_vec.emplace_back(t_nz);
        }
      } else {
        bool_indices_vec.emplace_back(input);
      }
    }
  }
  if (has_bool_mask) {
    c10::List<c10::optional<at::Tensor>> bool_mask_indices(bool_indices_vec);
    return bool_mask_indices;
  } else {
    return indices;
  }
}

static C10_UNUSED int hasContiguousSubspace(
    c10::ArrayRef<c10::IValue> indices_ival) {
  bool explicit_indices_together = false;
  int index_tensor_groups = 0;
  int index_tensor_group_start = 0;
  int dim = 0;
  for (auto input : indices_ival) {
    auto o1 = input.toOptional<at::Tensor>();
    if (o1.has_value() && !o1->defined()) {
      if (explicit_indices_together) {
        explicit_indices_together = false;
      }
    } else if (o1.has_value() && o1->defined()) {
      if (!explicit_indices_together) {
        index_tensor_group_start = dim;
        index_tensor_groups++;
      }
      explicit_indices_together = true;
    }
    dim++;
  }
  if (index_tensor_groups <= 1)
    return index_tensor_group_start;
  else
    return 0;
}

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
// transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
// tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
static C10_UNUSED std::tuple<at::Tensor, std::vector<c10::optional<at::Tensor>>>
transposeToFront(const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  c10::ArrayRef<c10::IValue> indices_ival = stack.at(1).toListRef();
  std::vector<int64_t> dims;
  std::vector<c10::optional<at::Tensor>> transposedIndices;
  std::vector<c10::optional<at::Tensor>> indices;
  for (const auto& index_opt : indices_ival) {
    auto o1 = index_opt.toOptional<at::Tensor>();
    if (o1.has_value() && o1.value().defined()) {
      const auto& index = o1.value();
      indices.emplace_back(std::move(index));
    } else {
      indices.emplace_back(c10::nullopt);
    }
  }
  dims.reserve(self.dim());
  for (const auto i : c10::irange(self.dim())) {
    if (indices[i].has_value()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    if (!indices[i].has_value()) {
      dims.push_back(i);
      transposedIndices.emplace_back(c10::nullopt);
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

static std::tuple<at::Tensor, std::vector<at::Tensor>>
generate_advanced_indexing_indices_list(const at::Stack& stack) {
  at::Tensor self = stack_tensor(stack, 0);
  c10::ArrayRef<c10::IValue> indices_ival = stack.at(1).toListRef();

  std::vector<c10::optional<at::Tensor>> indices;
  for (const auto& index_opt : indices_ival) {
    auto o1 = index_opt.toOptional<at::Tensor>();
    if (o1.has_value() && o1.value().defined()) {
      const auto& index = o1.value();
      indices.emplace_back(std::move(index));
    } else {
      indices.emplace_back(c10::nullopt);
    }
  }

  auto self_sizes = self.sizes().vec();
  std::vector<at::Tensor> indices_list;
  int64_t i = 0;
  int64_t index_t_sizes[self.dim()];
  bool index_all_elems[self.dim()];
  for (auto index_input : indices) {
    auto input = index_input;
    if (input.has_value() &&
        (input.value().scalar_type() != c10::ScalarType::Bool)) {
      index_all_elems[i] = false;
      index_t_sizes[i] = input.value().numel();
    } else if (!input.has_value()) {
      index_t_sizes[i] = self_sizes[i];
      index_all_elems[i] = true;
    }
    i++;
  }
  // account for any trailing dims that are not specified to be
  // indexed explicitly, but need to be taken care of.
  for (; i < self.dim(); i++) {
    index_all_elems[i] = true;
  }

  // Implement the repeat and repeat_interleaves logic so that the
  // indices, are interpreted as columns of the tuple which rows
  // become coordinates of the self tensor to update by values.
  // Each "explicit indice" shape is represented by the unique
  // sequence of indexes passed in the indice tensor for the given
  // dim of the self tensor. If there are two or more exactly the
  // same shapes in the indices list, they must share the same
  // repeat/repeat_interleave schema.
  // The above does not concern non-explicit indices. All appearances
  // of such indices should be adjusted with more repeats.
  // The algoritms starts with applying the repeat_interleave schema
  // for the first indice, and if for the next indice a new schema is
  // required the 1st repeat_interleave is replaced by the repeat and
  // so on. For example for four different indices:
  // self[2x1, :, 1x, 1x2]
  // where: 2x1 = [[0], [1]], 1x = [0], 1x2 = [0, 1]
  // the algorithm should:
  // 2x1 -> ri, ri, ri, ri
  // :   -> r,  ri, ri, ri
  // 1x  -> r,  r,  ri, ri
  // 1x2 -> r,  r,  r,  r
  // where ri is repeat_interleave and r is repeat
  // example indices data, where each of the columns
  // represets the repeated tensor data):
  // 2x1  :  1x  1x2
  //  0,  0,  0,  0   c0
  //  0,  0,  0,  1   c1
  //  0,  1,  0,  0   c2
  //  0,  1,  0,  1   c3
  //  1,  0,  0,  0   c4
  //  1,  0,  0,  1   c5
  //  1,  1,  0,  0   c6
  //  1,  1,  0,  1   c7
  // The above schema represents eight coordinates c0-7 to
  // update in the self tensor.
  // TODO:
  // adjust this algorithm to larger dim tensors as the r/ri logic
  // is applied dim wise and not on the flattened shape.
  int64_t repeats_needed[self.dim()];
  int64_t repeat_interleaves_needed[self.dim()];
  int repeat_index = 0;
  // Array holding a schape that was already processed through the r/ri logic
  // keep the index of the indice in order to copy schema if necessary from
  // repeats_needed and repeat_interleaves_needed arrays.
  std::vector<std::pair<std::vector<int64_t>, int64_t>> explicit_indice_handled;
  for (i = 0; i < self.dim(); i++) {
    repeats_needed[i] = 1;
    repeat_interleaves_needed[i] = 1;
    // Array required for saving shapes in order not to r/ri
    // if such operation was already performed on the saved shape.
    std::vector<std::vector<int64_t>> shapes_handled;

    if (!index_all_elems[i]) {
      auto shape_to_find = indices[i].value().sizes().vec();
      auto it = std::find_if(
          std::begin(explicit_indice_handled),
          std::end(explicit_indice_handled),
          [&](auto const& e) { return e.first == shape_to_find; });
      if (it != std::end(explicit_indice_handled)) {
        // If the r/ri schema was already performed for the
        // current indice shape, just copy it.
        repeats_needed[i] = repeats_needed[it->second];
        repeat_interleaves_needed[i] = repeat_interleaves_needed[it->second];
        continue;
      }
      // if the current indice is explicit, then save it in order
      // not to apply r/ri, if the same shape exists in the indice array input.
      shapes_handled.push_back(shape_to_find);
    }

    for (int j = 0; j < self.dim(); ++j) {
      if (i == j)
        continue;

      if (index_all_elems[j]) {
        if (repeat_index >= j)
          repeats_needed[i] *= self_sizes[j];
        else
          repeat_interleaves_needed[i] *= self_sizes[j];
      } else {
        // if the shape is explicit
        auto it = std::find(
            std::begin(shapes_handled),
            std::end(shapes_handled),
            indices[j].value().sizes().vec());
        if (it != std::end(shapes_handled))
          // and already handled, don't r/ri the i indice
          continue;

        shapes_handled.push_back(indices[j].value().sizes().vec());

        if (repeat_index >= j)
          repeats_needed[i] *= index_t_sizes[j];
        else
          repeat_interleaves_needed[i] *= index_t_sizes[j];
      }
    }

    if (!index_all_elems[i])
      // save the handled
      explicit_indice_handled.emplace_back(indices[i].value().sizes().vec(), i);

    // update the repeat index for any handled indice so one more repeat is
    // applied in the next iteration.
    ++repeat_index;
  }

  // Now create all indices tensors and insert into list using repeats_needed
  // and repeat_interleaves_needed. Note that in Boolean mask indexing method we
  // have to create indices from the mask using nonzero and squeeze as done in
  // previous code block.
  for (int dim = 0; dim < self.dim(); dim++) {
    at::Tensor it;
    if (index_all_elems[dim]) {
      std::vector<int64_t> shape{self.sizes().vec()[dim]};
      c10::IntArrayRef arange_size(shape.data(), shape.size());
      at::TensorOptions options =
          self.options().dtype(c10::ScalarType::Long).device(c10::kHPU);
      auto generated_index_tensor =
          habana_lazy::empty_hpu_lazy(arange_size, options, c10::nullopt);
      generated_index_tensor = at::arange(
          0,
          self.sizes().vec()[dim],
          1,
          c10::ScalarType::Long,
          c10::nullopt,
          c10::kHPU,
          c10::nullopt);
      it = generated_index_tensor;
      auto it_repeat_interleave =
          it.repeat_interleave(repeat_interleaves_needed[dim]);
      indices_list.push_back(it_repeat_interleave.repeat(repeats_needed[dim]));
    } else if (indices[dim].has_value()) {
      auto input_temp = indices[dim].value();
      Tensor in_t;
      if (input_temp.defined() &&
          (input_temp.device().type() != c10::DeviceType::HPU)) {
        in_t = input_temp.to(c10::kHPU);
      } else {
        in_t = input_temp;
      }
      // auto input = indices[dim];
      // at::Tensor in_t = input.value();
      auto it_repeat_interleave =
          in_t.repeat_interleave(repeat_interleaves_needed[dim]);
      indices_list.push_back(it_repeat_interleave.repeat(repeats_needed[dim]));
    }
  }
  // return indices_list;
  return std::make_tuple(self, std::move(indices_list));
}

Tensor& _index_put_impl_hpu_lazy_(
    Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices_in,
    const Tensor& value,
    bool accumulate,
    [[maybe_unused]] const bool unsafe) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;
  c10::List<c10::optional<at::Tensor>> indices;
  bool advanced_indexing = check_for_advanced_indexing(indices_in);
  if (advanced_indexing) {
    // if we have boolean mask tensors, convert them to long int indices
    indices = check_for_boolean_advanced_indexing(indices_in);
  } else {
    indices = indices_in;
  }
  std::vector<at::Tensor> indices_vec;
  TORCH_CHECK(
      self.dim() <= MAX_DIMS_FOR_ADVANCED_INDEXING,
      "index_put op doesn't support more than ",
      MAX_DIMS_FOR_ADVANCED_INDEXING,
      " dims");
  at::Tensor self_permuted;
  if (self.device().type() != c10::DeviceType::HPU)
    self_permuted = self.to(c10::kHPU);
  else
    self_permuted = self;

  at::Tensor value_in;
  if (value.device().type() != c10::DeviceType::HPU)
    value_in = value.to(c10::kHPU);
  else
    value_in = value;
  if (advanced_indexing) {
    at::Stack stack;
    stack.emplace_back(self);
    stack.emplace_back(c10::IValue(indices));
    std::tie(self_permuted, indices_vec) =
        generate_advanced_indexing_indices_list(stack); //(self, indices_in);
  } else {
    for (c10::optional<at::Tensor> input_ind : indices) {
      auto input_temp = input_ind.value_or(Tensor());
      at::Tensor input;
      if (input_temp.defined() &&
          (input_temp.device().type() != c10::DeviceType::HPU)) {
        input = input_temp.to(c10::kHPU);
      } else {
        input = input_temp;
      }
      if (input.defined()) {
        indices_vec.push_back(input);
      } else {
        HABANA_ASSERT(
            0 &&
            "index_put: unsupported case: None is not yet supported on HPU for c10::List<c10::optional<Tensor>>");
      }
    }
  }

  // handle views for tensorlist indices
  at::TensorList indices_in_list(indices_vec);
  auto indices_out_vec =
      habana_lazy::HbLazyTensorViews::HandleViewsTensorList(indices_in_list);
  std::vector<c10::optional<at::Tensor>> indices_out_opt_vec;
  for (auto ind : indices_out_vec) {
    if (ind.defined()) {
      indices_out_opt_vec.emplace_back(ind);
    } else {
      indices_out_opt_vec.emplace_back(c10::nullopt);
    }
  }
  c10::List<c10::optional<at::Tensor>> indices_out_opt_list(
      indices_out_opt_vec);
  // index backward is not supported on hpu, indices needs to be
  // bool, byte or long type for cpu fallback
  index_put_hpu_lazy_(
      self_permuted, indices_out_opt_list, value_in, accumulate);
  return self;
}

Tensor slice_shape_tensor(const Tensor& shape_tensor) {
  std::vector<at::IValue> vector_of_inputs = {shape_tensor, 0, 1};
  auto end_shape = DimVector{1};

  using T = at::Tensor;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(const std::vector<at::IValue>& vector_of_inputs)
        : LazyOp<T>("aten::select", vector_of_inputs, nullptr, -1) {}

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
  habana_lazy::NoAccThread no_acc_thread;

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
    flush_op(1);
    return output;
  }

  using T = std::tuple<at::Tensor, at::Tensor>;
  struct NonZero : LazyOp<T> {
    explicit NonZero(
        const std::vector<at::IValue>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor>>(
              "hpu::nonzero",
              inputs,
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
  NonZero k({self, c10::nullopt}, {output_shape, shape_tensor_shape});
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
    flush_op(1);
    return output;
  }

  // Add a slice node to capture relevent elements from nonzero node
  // in case we have relevant elements
  auto result = slice_hpu_lazy(where_tensor, 0, 0, end, 1);
  flush_op(1);
  return result;
}

Tensor& nonzero_out_hpu_lazy(const Tensor& self, Tensor& output) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

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
    flush_op(1);
    return output;
  }

  // Add nonzero node
  std::vector<int64_t> output_shape{elements, dimensions};
  std::vector<int64_t> shape_tensor_shape{5};
  using T = std::tuple<at::Tensor, at::Tensor>;
  LazyOp<T> k(
      "hpu::nonzero",
      {self, c10::nullopt},
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
  hl_end.IrSetNode(node_slice);
  // Force an exections here to capture second element of shape tensor.
  // This element is required to determine shape of next node's output
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
    flush_op(1);
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

  hl_result.IrSetNode(node);
  flush_op(1);
  return output;
}

std::vector<at::Tensor> unbind_hpu_lazy_(const Tensor& self, int64_t dim) {
  PT_LAZY_TRACE;

  return at::native::unbind(self, dim);
}

Tensor masked_select_hpu_lazy(const Tensor& self, const Tensor& mask) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  // return input if the dimension of the mask is greater than the dimension of
  // input tensor.
  if (mask.dim() > self.dim()) {
    return self;
  }

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
  habana_lazy::NoAccThread no_acc_thread;

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

  handle_collective(self);
  handle_collective(indices);
  handle_collective(source);
  handle_collective(out);

  auto func = [self, dim, indices, source, alpha, out]() mutable {
    auto dim_ = at::maybe_wrap_dim(dim, self.dim(), true);

    auto dim_size = self.numel(); // for scalar tensor case
    if (!self.sizes().vec().empty()) {
      // for non-scalar tensor case
      dim_size = self.sizes().vec()[dim_];
    }

    std::string op_name = "aten::index_add";
    if (indices.numel() > dim_size) {
      // Implementation to take care of duplicate entries in index tensor and
      // also the case where index tensor size can be greater than the self
      // tensor size at the relevant dim. For now enable this only in the large
      // index tensor condition. This will later be enabled as default after
      // watching for perf impacts.
      op_name = "hpu::index_add";
    }
    LazyOp<Tensor> index_add_op(
        op_name, {self, dim_, indices, source, alpha}, {self.sizes().vec()}
        // out_shapes
    );

    Tensor index_add_out = index_add_op.call();

    LazyOp<at::Tensor&> k{"hpu::habana_d2d_memcpy_other", {index_add_out, out}};
    // Can't call get_result, because sizes were set in the main thread and
    // inner if will evalueate to false. So I need to set proper flag for shape
    // change manually.
    k.set_shape_changed();

    return k.call(out);
  };

  out.unsafeGetTensorImpl()->set_sizes_contiguous(self.sizes());

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(index_add_out, func, out)
}

Tensor& index_add_hpu_lazy_(
    Tensor& self,
    int64_t dim,
    const Tensor& indices,
    const Tensor& source,
    const Scalar& alpha) {
  PT_LAZY_TRACE;

  handle_collective(self);
  handle_collective(indices);
  handle_collective(source);

  auto func = [self, dim, indices, source, alpha]() mutable {
    // TPC doesn't support inplace index add natively
    // Implement using out of place index add followed by D2D copy
    // TODO revisit once strided mem copy feature is mature
    auto dim_ = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
    auto hl_self = GetOrCreateHbLazyTensor(self);

    auto dim_size = self.numel(); // for scalar tensor case
    if (!self.sizes().vec().empty()) {
      // for non-scalar tensor case
      dim_size = self.sizes().vec()[dim_];
    }

    std::string op_name = "aten::index_add";
    if (indices.numel() > dim_size) {
      // Implementation to take care of duplicate entries in index tensor and
      // also the case where index tensor size can be greater than the self
      // tensor size at the relevant dim. For now enable this only in the large
      // index tensor condition. This will later be enabled as default after
      // watching for perf impacts.
      op_name = "hpu::index_add";
    }
    LazyOp<Tensor> index_add_op(
        op_name, {self, dim_, indices, source, alpha}, {self.sizes().vec()}
        // out_shapes
    );

    Tensor index_add_out = index_add_op.call();

    LazyOp<at::Tensor&> k{
        "hpu::habana_d2d_memcpy_other", {index_add_out, self}};
    return k.call(self);
  };

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(index_add_, func, self)
}

bool static cast_required(c10::ScalarType self_scalar_type) {
  if (self_scalar_type != c10::ScalarType::Double &&
      self_scalar_type != c10::ScalarType::Float &&
      self_scalar_type != c10::ScalarType::Half &&
      self_scalar_type != c10::ScalarType::BFloat16 &&
      self_scalar_type != c10::ScalarType::Long &&
      self_scalar_type != c10::ScalarType::Int) {
    return true;
  }
  return false;
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
    PT_IRGRAPH_DEBUG("step marker due to index_put_frontend_impl_hpu_lazy");
    HbLazyTensor::StepMarker({});
    for (size_t i = 0; i < indices_vec.size(); i++) {
      auto list = torch::nonzero_numpy(indices_vec.at(i));
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
    flush_op(1);
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

  if (cast_required(self.scalar_type())) {
    // i8/i16/i32 -> f32
    LazyOp<at::Tensor> k_{
        "hpu::cast", {self, c10::ScalarType::Int}, {self.sizes().vec()}};
    k_.set_scalar_types({c10::ScalarType::Int});
    self_cast = k_.call();
    // i8/i16/i32 -> f32
    LazyOp<at::Tensor> kv_{
        "hpu::cast",
        {value_in, c10::ScalarType::Int},
        {value_in.sizes().vec()}};
    kv_.set_scalar_types({c10::ScalarType::Int});
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
    if (cast_required(self.scalar_type())) {
      LazyOp<at::Tensor> k_{
          "hpu::cast",
          {scatter_nd_out, self.scalar_type()},
          {scatter_nd_out.sizes().vec()}};
      k_.set_scalar_types({self.scalar_type()});
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

    if (cast_required(self.scalar_type())) {
      LazyOp<at::Tensor> k_{
          "hpu::cast", {result, self.scalar_type()}, {result.sizes().vec()}};
      k_.set_scalar_types({self.scalar_type()});
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
    flush_op(1);
    return {output, output};
  }

  using T = std::tuple<at::Tensor, at::Tensor>;
  struct NonZero : LazyOp<T> {
    explicit NonZero(
        const std::vector<at::IValue>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor>>(
              "hpu::nonzero",
              inputs,
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
      flush_op(2);
      return {where_tensor, shape_tensor};
    }
  };

  // Add nonzero node
  auto output_shape = NonZeroOperator::compute_output_shape(self);
  std::vector<int64_t> shape_tensor_shape{5};
  Tensor nz_shape_tensor;
  c10::optional<at::Tensor> nonzero_shape_tensor =
      c10::make_optional(nz_shape_tensor);
  NonZero k({self, c10::nullopt}, {output_shape, shape_tensor_shape});
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
  habana_lazy::NoAccThread no_acc_thread;

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
      flush_op(1);
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
  if (habana_helpers::GetRefineDynamicShapeStatus() ||
      GET_ENV_FLAG_NEW(PT_HPU_FORCE_INDEX_PUT_FRONTEND_FALLBACK)) {
    return index_put_frontend_impl_hpu_lazy(
        self, indices_list, value_in, accumulate);
  }
  LazyOp<at::Tensor> index_put_op{
      "hpu::index_put_normal_and_neg_indices",
      {self, indices, value_in, accumulate}};
  return index_put_op.call();
}

Tensor& index_put_hpu_lazy_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    bool accumulate) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  std::vector<at::Tensor> indices_in;
  for (c10::optional<Tensor> input : indices) {
    indices_in.push_back(input.value());
  }

  auto isIndicesBool = indices_in[0].scalar_type() == c10::ScalarType::Bool;
  auto self_clone = self;
  auto index_put_result =
      index_put_hpu_lazy(self_clone, indices, value, accumulate);

  auto& stride_params_opt = GetHbLazyTensor(self).getDataPtr()->stride_params;
  if (!stride_params_opt.has_value()) {
    LazyOp<at::Tensor&> k{
        "hpu::habana_d2d_memcpy_other", {index_put_result, self}};
    self = k.call(self);
  } else {
    HbLazyTensorViews::HandleViewsD2D(index_put_result, self);
  }

  // In DS case changing shapes will not cause a cache miss, therefore no need
  // to break index_put op from subsequent graph whereas in other cases
  // changing shapes will cause cache misses therefore breaking graph.
  if (GET_ENV_FLAG_NEW(PT_HPU_FORCE_INDEX_PUT_FRONTEND_FALLBACK) ||
      (!habana_helpers::GetRefineDynamicShapeStatus() && isIndicesBool &&
       (accumulate || value.dim()))) {
    std::vector<HbLazyTensor> hl_flush_end = {GetHbLazyTensor(self)};
    HbLazyTensor::SyncTensorsGraph(&hl_flush_end);
  } else {
    flush_op(1);
  }
  return self;
}

Tensor slice_hpu_lazy(
    const Tensor& self_in,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  PT_LAZY_TRACE;

  // Native fork implementation to slice op is introduced to
  // allocate correct autograd gradient function for view tensor.
  auto out = at::native::slice(self_in, dim, start, end, step);

  // lazy eager optimized view handling (no need to create view table)
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) == true)) {
    return out;
  }

  auto param_setter = [dim, start, end, step](
                          const Tensor& self_in, StrideParams& strided_param) {
    strided_param.optype = kStridedOpSlice;
    int64_t start_ = start.has_value() ? start.value() : 0;
    int64_t end_ = end.has_value() ? end.value() : INT64_MAX;
    StridedOpSliceParams slice_param = {dim, start_, end_, step};
    strided_param.params.slice_param = slice_param;
    PT_VIEWTABLE_DEBUG(
        "slice fallback tensor id ",
        GetHbLazyTensorId(self_in),
        " dim ",
        dim,
        " start ",
        start_,
        " end ",
        end_,
        " step ",
        step);
  };
  RUN_VIEW_OP_MAYBE_WITH_ACC_THREAD(slice, self_in, out, param_setter);
}

Tensor slice_backward_hpu_lazy(
    const Tensor& grad_output,
    SymIntArrayRef input_sizes_sym,
    int64_t dim,
    c10::SymInt start_sym,
    c10::SymInt end_sym,
    c10::SymInt step_sym) {
  PT_LAZY_TRACE;

  IntArrayRef input_sizes = c10::asIntArrayRefUnchecked(input_sizes_sym);
  int64_t start = start_sym.expect_int();
  int64_t end = end_sym.expect_int();
  int64_t step = step_sym.expect_int();

  const auto grad_output_sizes = grad_output.sizes();
  if (std::find(grad_output_sizes.begin(), grad_output_sizes.end(), 0) !=
      grad_output_sizes.end()) {
    return torch::zeros(input_sizes, at::TensorOptions(at::kHPU));
  } else {
    return at::native::slice_backward(
        grad_output, input_sizes, dim, start, end, step);
  }
}

Tensor alias_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  auto out = as_strided_hpu_lazy(
      self, self.sizes(), self.strides(), self.storage_offset());

  // lazy eager optimized view handling (no need to create view table)
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) == true)) {
    return out;
  }

  auto param_setter = [](const Tensor& self, StrideParams& strided_param) {
    strided_param.optype = kStridedOpIdentity;

    PT_VIEWTABLE_DEBUG(
        "alias-identity fallback tensor id ", GetHbLazyTensorId(self));
  };
  RUN_VIEW_OP_MAYBE_WITH_ACC_THREAD(alias, self, out, param_setter);
}

at::Tensor select_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    c10::SymInt index_sym) {
  PT_LAZY_TRACE;
  int64_t ndim = self.dim();
  int64_t index = index_sym.expect_int();
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

Tensor kl_div_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k(
      "aten::kl_div",
      {self, target, reduction, log_target},
      {KlDivOperator::compute_output_shape(self, reduction)});
  RUN_MAYBE_WITH_ACC_THREAD(kl_div, k)
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
  // TPC  BN supports only 4D inputs. This means that any higher dims have to
  // be flattened
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
        : LazyOp<T>("hpu::native_batch_norm_training", inputs, nullptr, -1) {}
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
      RUNNING_HASH_COMBINE_OUTPUT_TENSOR(result_img);
      RUNNING_HASH_COMBINE_OUTPUT_TENSOR(result_mean);
      RUNNING_HASH_COMBINE_OUTPUT_TENSOR(result_var);
      RUNNING_HASH_COMBINE_OUTPUT_TENSOR(running_mean);
      RUNNING_HASH_COMBINE_OUTPUT_TENSOR(running_var);
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
  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(native_batch_norm_training, op)
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
  RUN_MAYBE_WITH_ACC_THREAD(native_batch_norm_inf, op)
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

std::tuple<Tensor, Tensor, Tensor> batch_norm_legit_hpu_lazy(
    const Tensor& input_,
    const c10::optional<at::Tensor>& weight_tensor,
    const c10::optional<at::Tensor>& bias_tensor,
    at::Tensor& running_mean_,
    at::Tensor& running_var_,
    bool training,
    double momentum,
    double eps) {
  PT_LAZY_TRACE;
  return batch_norm_hpu_lazy(
      input_,
      weight_tensor,
      bias_tensor,
      running_mean_,
      running_var_,
      training,
      momentum,
      eps);
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
        : LazyOp<T>("hpu::native_batch_norm_backward", inputs, nullptr, -1) {}
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
  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(native_batch_norm_backward, op)
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
    [[maybe_unused]] std::array<bool, 3> output_mask) {
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
    const ::std::optional<at::Tensor>& weight,
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

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_group_norm_backward_hpu_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& input_,
    const at::Tensor& mean_,
    const at::Tensor& rstd_,
    const c10::optional<at::Tensor>& weight_opt,
    [[maybe_unused]] c10::SymInt N,
    [[maybe_unused]] c10::SymInt C,
    [[maybe_unused]] c10::SymInt HxW,
    int64_t num_groups,
    [[maybe_unused]] std::array<bool, 3> output_mask) {
  if (input_.numel() == 0) {
    auto output = at::empty_like(
        input_, input_.options(), input_.suggest_memory_format());
    auto grad_gamma =
        at::zeros(c10::asIntArrayRefUnchecked({C}), input_.options());
    auto grad_beta =
        at::zeros(c10::asIntArrayRefUnchecked({C}), input_.options());
    return std::make_tuple(output, grad_gamma, grad_beta);
  }
  int64_t Nmod = N.expect_int() * num_groups;
  // ================= BEGIN :Re create BN FWD output from BN FWD input
  // ================== This can be avoided only if autograd override is done
  // and BN FWD o/p is stored for use in BWD
  bool use_bn_fwd_in_gn_bwd = GET_ENV_FLAG_NEW(PT_HPU_USE_BN_FWD_IN_GN_BWD);
  at::Tensor mean;
  at::Tensor rstd;
  // batch_norm_bwd TPC guid expects mean_istd tensor for stage2 as Float
  // so, whatever the fwd generated as mean/istd based on input Float/BF16,
  // we need to convert mean_istd to Float
  mean = mean_.to(c10::ScalarType::Float);
  rstd = rstd_.to(c10::ScalarType::Float);

  Tensor bn_fwd_out;
  auto input_shape = input_.sizes().vec();
  int64_t rszarr_bn_in[input_.dim()];
  int64_t rszarr_bn_fwd_mean[input_.dim()];
  int64_t m = input_.numel() / Nmod;
  for (int i = 0; i < input_.dim(); i++) {
    rszarr_bn_in[i] = 1;
    rszarr_bn_fwd_mean[i] = 1;
  }
  rszarr_bn_in[1] = Nmod;
  rszarr_bn_in[input_.dim() - 1] = m;
  rszarr_bn_fwd_mean[1] = Nmod;
  c10::IntArrayRef bn_in_view_shape(rszarr_bn_in, input_.dim());
  auto bn_fwd_in = at::reshape(
      input_, bn_in_view_shape); // view_hpu(input, bn_in_view_shape);
  if (use_bn_fwd_in_gn_bwd) {
    Tensor bn_wt1, bn_bt1, bn_rm1, bn_rv1;
    auto x = batch_norm_hpu_lazy(
        bn_fwd_in, bn_wt1, bn_bt1, bn_rm1, bn_rv1, true, 0.001, 1e-5);
    auto bn_fwd_out_tmp = std::get<0>(x);
    bn_fwd_out = at::reshape(bn_fwd_out_tmp, input_shape);
  } else {
    c10::IntArrayRef mean_for_bn_fwd_shape(rszarr_bn_fwd_mean, input_.dim());
    auto mean_for_bn_fwd = at::reshape(mean, mean_for_bn_fwd_shape);
    auto rstd_for_bn_fwd = at::reshape(rstd, mean_for_bn_fwd_shape);
    auto bn_fwd_out_tmp =
        at::mul(at::sub(bn_fwd_in, mean_for_bn_fwd), rstd_for_bn_fwd);
    bn_fwd_out = at::reshape(bn_fwd_out_tmp, input_shape);
  }

  int64_t dimarr[input_.dim() - 1];
  for (int i = 0; i < (input_.dim() - 1); i++)
    dimarr[i] = i + 1;
  dimarr[0] = 0;
  c10::IntArrayRef reduce_dims(dimarr, input_.dim() - 1);

  auto t1 = at::mul(grad_out, bn_fwd_out);

  int64_t rszarr1[input_.dim()];
  for (int i = 0; i < input_.dim(); i++)
    rszarr1[i] = 1;
  rszarr1[1] = C.expect_int();
  c10::IntArrayRef wt_view_shape(rszarr1, input_.dim());
  auto weight = weight_opt.value_or(Tensor());
  if (!weight.defined()) {
    auto options = torch::TensorOptions()
                       .dtype(c10::ScalarType::Float)
                       .device(torch::kHPU)
                       .requires_grad(false);
    weight = torch::ones(wt_view_shape.vec(), options);
  }
  auto grad_beta = at::sum(grad_out, reduce_dims, false).to(weight.dtype());
  auto grad_gamma = at::sum(t1, reduce_dims, false).to(weight.dtype());

  auto weight_view = at::reshape(weight, wt_view_shape);
  auto t2 = at::mul(grad_out, weight_view);

  auto t3 = at::reshape(t2, bn_in_view_shape);
  Tensor bn_gout;
  if (grad_out.scalar_type() != input_.scalar_type()) {
    bn_gout = t3.to(input_.scalar_type());
  } else {
    bn_gout = t3;
  }
  auto bn_in = at::reshape(input_, bn_in_view_shape);

  std::vector<int64_t> view_mean_shape{Nmod};
  auto mean_reshaped = at::reshape(mean, view_mean_shape);
  auto rstd_reshaped = at::reshape(rstd, view_mean_shape);

  Tensor bn_wt, bn_rm, bn_rv; // Undefined
  std::array<bool, 3> bn_output_mask = {true, false, false};
  auto g = batch_norm_bwd_hpu_lazy(
      bn_gout,
      bn_in,
      bn_wt,
      bn_rm,
      bn_rv,
      mean_reshaped,
      rstd_reshaped,
      true,
      1e-05,
      bn_output_mask);
  auto gin = at::reshape(std::get<0>(g), grad_out.sizes());

  return std::make_tuple(gin, grad_gamma, grad_beta);
}

std::tuple<Tensor, Tensor, Tensor> instance_norm_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    double eps) {
  PT_LAZY_TRACE;

  auto mean_var_shape = {
      input.sizes().vec()[INPUT_BATCH_INDEX],
      input.sizes().vec()[INPUT_CHANNEL_INDEX]};

  using T = std::tuple<Tensor, Tensor, Tensor>;
  LazyOp<T> k(
      "hpu::instance_norm",
      {input, weight, bias, eps},
      {input.sizes().vec(), mean_var_shape, mean_var_shape} // out_shapes
  );

  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(instance_norm, k)
}

std::tuple<Tensor, Tensor, Tensor> instance_norm_backward_hpu_lazy(
    const Tensor& input,
    const Tensor& grad_in,
    const Tensor& mean,
    const Tensor& istd,
    const Tensor& gamma) {
  PT_LAZY_TRACE;

  auto grad_beta_gamma_shape = {input.sizes().vec()[INPUT_CHANNEL_INDEX]};
  using T = std::tuple<Tensor, Tensor, Tensor>;
  LazyOp<T> k(
      "hpu::instance_norm_backward",
      {input, grad_in, mean, istd, gamma},
      {input.sizes().vec(), grad_beta_gamma_shape, grad_beta_gamma_shape});

  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(instance_norm_backward, k)
}

at::Tensor& randperm_hpu_lazy_ht(Tensor& output, int64_t n, at::Tensor seed) {
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
  auto tmeta{get_tensor_extra_meta(hl_param_internal)};
  tmeta->set_host_data(
      params_vec.data(),
      params_vec.size(),
      sizeof(int32_t),
      HostDataType::INT32_T);
  LazyOp<Tensor&> hpu_op{
      "hpu::randperm_out_ds_ht", {params_shape, seed, output}, nullptr, 2};
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(randperm_hpu_lazy_ht, hpu_op, output);
}

void setTensorDim(at::Tensor& tensor, int64_t n) {
  auto hl_result = GetOrCreateHbLazyTensor(tensor, c10::kHPU);
  auto shape = DimVector({n});
  auto reshaped = hl_result.getAttachedTensorImpl();
  THHTensor_resizeNd(reshaped, shape.size(), shape.data(), nullptr);
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(shape));
}
#if IS_PYTORCH_OLDER_THAN(2, 1)
Tensor& randperm_hpu_lazy(
    int64_t n,
    c10::optional<Generator> gen,
    Tensor& output) {
  PT_LAZY_TRACE;
  auto seed = habana::get_seed_tensor_hpu(gen);
  // resizing the output as it is coming as empty from model
  setTensorDim(output, n);

  // Currently synapse support dynamic shape arange only for int datatypes.
  // For any other output datatype, will fallback to normal flow.
  if (habana_helpers::GetRefineDynamicShapeStatus() &&
      (output.scalar_type() == c10::ScalarType::Int ||
       output.scalar_type() == c10::ScalarType::Long)) {
    return randperm_hpu_lazy_ht(output, n, seed);
  } else {
    LazyOp<Tensor&> op{
        "hpu::randperm_out", {Scalar((int32_t)n), seed, output}, {{n}}};
    RUN_INPLACE_MAYBE_WITH_ACC_THREAD(randperm_hpu_lazy_ht, op, output);
  }
}
#else
Tensor& randperm_hpu_lazy(
    c10::SymInt n_,
    c10::optional<Generator> gen,
    Tensor& output) {
  PT_LAZY_TRACE;
  auto n = n_.expect_int();
  auto seed = habana::get_seed_tensor_hpu(gen);
  // resizing the output as it is coming as empty from model
  setTensorDim(output, n);

  // Currently synapse support dynamic shape arange only for int datatypes.
  // For any other output datatype, will fallback to normal flow.
  if (habana_helpers::GetRefineDynamicShapeStatus() &&
      (output.scalar_type() == c10::ScalarType::Int ||
       output.scalar_type() == c10::ScalarType::Long)) {
    return randperm_hpu_lazy_ht(output, n, seed);
  } else {
    LazyOp<Tensor&> op{
        "hpu::randperm_out", {Scalar((int32_t)n), seed, output}, {{n}}};
    RUN_INPLACE_MAYBE_WITH_ACC_THREAD(randperm_hpu_lazy_ht, op, output);
  }
}

Tensor randperm_nogen_hpu_lazy(
    c10::SymInt n,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  PT_LAZY_TRACE;
  at::TensorOptions options = at::TensorOptions()
                                  .device(device)
                                  .dtype(c10::ScalarType::Int)
                                  .layout(layout)
                                  .pinned_memory(pin_memory);
  std::vector<int64_t> out_size{n.expect_int()};
  auto out_t =
      empty_hpu_lazy(out_size, options, c10::MemoryFormat::Contiguous, true);
  out_t = randperm_hpu_lazy(n, c10::nullopt, out_t);
  return out_t.to(dtype.value_or(c10::ScalarType::Int));
}
#endif
at::Tensor repeat_hpu_lazy_ht(const at::Tensor& self, at::IntArrayRef repeats) {
  PT_LAZY_TRACE;
  std::vector<at::IValue> vector_of_inputs;
  std::string op_name;

  // Handle 0D tensor
  if (self.sizes().empty() && self.numel() == 1 && repeats.empty()) {
    return self;
  }

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

  auto tmeta{get_tensor_extra_meta(hl_param_internal)};
  tmeta->set_host_data(
      params_vec.data(),
      params_vec.size(),
      sizeof(int32_t),
      HostDataType::INT32_T);
  tmeta->set_H2D_data_for_bucketing();
  auto out_shape = RepeatOperator::compute_output_shape(self, repeats);

  vector_of_inputs = {self, params_shape};
  op_name = "hpu::repeat_ht";
  LazyOp<at::Tensor> k{op_name, vector_of_inputs, {out_shape}};
  RUN_MAYBE_WITH_ACC_THREAD(repeat, k)
}

at::Tensor repeat_hpu(const at::Tensor& self, at::SymIntArrayRef _repeats) {
  PT_LAZY_TRACE;
  auto repeats = C10_AS_INTARRAYREF_SLOW(_repeats);
  if (habana_helpers::GetRefineDynamicShapeStatus()) {
    return repeat_hpu_lazy_ht(self, repeats);
  }

  LazyOp<at::Tensor> k{
      "aten::repeat",
      {self, repeats},
      {RepeatOperator::compute_output_shape(self, repeats)}};
  RUN_MAYBE_WITH_ACC_THREAD(repeat, k)
}

at::Tensor repeat_inlv_hpu_lazy(
    const at::Tensor& repeats,
    c10::optional<int64_t> output_size) {
  // if output_size is not provided by user, there is no way to compute output
  // shape without peeking into the "repeats" tensor. See desc. from PyT docs,
  // "output_size (int, optional) – Total output size for the given axis (
  // e.g. sum of repeats). If given, it will avoid stream syncronization
  // needed to calculate output shape of the tensor."

  // In our case because of the use of H2D tensor for repeats, we will always
  // break the graph if model puts repeats tensor on HPU, but this should be
  // ok as this will not cause a blocking synchronization
  int64_t out_size;
  // repeats can only by "long" or "int", if long, cast to int because synapse
  // cannot handle long tensors
  bool need_h2d_tensor = false;
  if (habana_helpers::GetRefineDynamicShapeStatus() ||
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
  auto input = at::arange(
      repeats.sizes()[0],
      c10::optTypeMetaToScalarType(repeats.options().dtype_opt()),
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
    auto tmeta{get_tensor_extra_meta(hl_param_internal)};
    tmeta->set_host_data(
        repeats_cpu.data_ptr(),
        repeats_cpu.sizes()[0],
        sizeof(int32_t),
        HostDataType::INT32_T);
    tmeta->set_H2D_data_for_bucketing();

    auto output_shape =
        RepeatInlvOperator::compute_output_shape(input, 0, out_size);
    LazyOp<at::Tensor> k{
        "hpu::repeat_inlv_ht", {input, repeats_tensor, 0}, {output_shape}};
    return k.call();
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
  auto tmeta{get_tensor_extra_meta(hl_output_shape_tensor_internal, true)};
  if (tmeta) {
    tmeta->set_H2D_frontend_shape_tensor();
  }

  LazyOp<at::Tensor> k{
      "hpu::repeat_inlv",
      {input, repeats, 0, output_shape_tensor},
      {output_shape}};
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

Tensor empty_strided_hpu_lazy(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options,
    bool create_storage,
    synTensorType tensor_type,
    int64_t storage_offset,
    c10::optional<std::reference_wrapper<const at::Tensor>> base_view,
    bool is_strided) {
  PT_LAZY_TRACE;

  at::Tensor empty_tensor = empty_hpu_lazy(
      size,
      options,
      c10::nullopt,
      create_storage,
      tensor_type,
      base_view,
      is_strided);
  empty_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

  if (storage_offset) {
    empty_tensor.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  }

  // lazy eager optimized view handling
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) == true) &&
      is_strided) {
    return empty_tensor;
  }

  // empty_hpu_lazy call might move the tensor to cpu for unsupported dtypes
  if (empty_tensor.device().type() != c10::DeviceType::HPU)
    return empty_tensor;
  // If we have created a tensor with storage, set the strides and sizes to
  // backend tensor as well
  if (create_storage) {
    auto hl_empty = TryGetHbLazyTensor(empty_tensor);
    if (hl_empty) {
      hl_empty.value().IrInitAsInputNode();
    }
  }
  return empty_tensor;
}

Tensor transpose_hpu_lazy(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  PT_LAZY_TRACE;

  auto out = at::native::transpose(self, dim0_, dim1_);

  // lazy eager optimized view handling (no need to create view table)
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) == true)) {
    return out;
  }

  // To Do - to check if below handling required in lazye eager.
  // if due to any reason 'out' does not have the storage then we
  // need to attach the storage of 'self' to 'out'

  // at::native::transpose can return back self w/o invoking as_strided under
  // certain cases like 1D/dim0 == dim1. Skip view table access in such cases
  auto additional_predicate = [](const Tensor& self, const Tensor& out) {
    auto self_id = GetHbLazyTensorId(self);
    auto out_id = GetHbLazyTensorId(out);
    return out_id != self_id;
  };
  auto param_setter = [dim0_, dim1_](
                          const Tensor& self, StrideParams& strided_param) {
    strided_param.optype = kStridedOpTranspose;
    StridedOpTransposeParams transpose_param = {dim0_, dim1_};
    strided_param.params.transpose_param = transpose_param;

    PT_VIEWTABLE_DEBUG(
        "transpose fallback tensor id ",
        GetHbLazyTensorId(self),
        " dim0 ",
        dim0_,
        " dim1 ",
        dim1_);
  };
  RUN_WITH_PREDICATE_VIEW_OP_MAYBE_WITH_ACC_THREAD(
      transpose, self, out, param_setter, additional_predicate);

  return out;
}

Tensor t_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  if (self.dim() < 2) {
    return self;
  }
  auto out = at::native::t(self);

  // lazy eager optimized view handling (no need to create view table)
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) == true)) {
    return out;
  }

  // To Do - To check if any special handling required for
  // 0-D and 1-D input for lazy eager

  auto param_setter = [](const Tensor& self, StrideParams& strided_param) {
    strided_param.optype = kStridedOpT;
    PT_VIEWTABLE_DEBUG("t fallback tensor id ", GetHbLazyTensorId(self));
  };

  RUN_VIEW_OP_MAYBE_WITH_ACC_THREAD(t, self, out, param_setter);
}

Tensor squeeze_hpu_lazy(const Tensor& self, int64_t dim_) {
  PT_LAZY_TRACE;

  auto dim = dim_;
  Tensor out;

  if (dim != HABANA_DIM_MAX) {
    dim = at::maybe_wrap_dim(dim_, self.dim());
    out = at::native::squeeze(self, dim);
  } else {
    out = at::native::squeeze(self);
  }

  // lazy eager optimized view handling (no need to create view table)
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) == true)) {
    return out;
  }

  auto param_setter = [dim](const Tensor& self, StrideParams& strided_param) {
    strided_param.optype = kStridedOpSqueeze;
    StridedOpSqueezeParams squeeze_param = {dim};
    strided_param.params.squeeze_param = squeeze_param;

    PT_VIEWTABLE_DEBUG(
        "squeeze fallback tensor id ", GetHbLazyTensorId(self), " dim ", dim);
  };

  RUN_VIEW_OP_MAYBE_WITH_ACC_THREAD(squeeze, self, out, param_setter);
}

Tensor squeeze_self_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  // using invalid dim size HABANA_DIM_MAX to signal the backend kernel that
  // squeeze needs to be performed on all applicable axes
  return squeeze_hpu_lazy(self, HABANA_DIM_MAX /*dim*/);
}

Tensor squeeze_dim_hpu_lazy(const Tensor& self, int64_t dim) {
  PT_LAZY_TRACE;

  return squeeze_hpu_lazy(self, dim);
}

Tensor squeeze_dims_hpu_lazy(const Tensor& self, IntArrayRef dims) {
  PT_LAZY_TRACE;

  Tensor out;
  auto dims_vec = dims.vec();
  at::wrap_all_dims(dims_vec, self.dim());
  out = at::native::squeeze(self, dims_vec);

  // lazy eager optimized view handling (no need to create view table)
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) == true)) {
    return out;
  }

  auto param_setter = [dims_vec](
                          const Tensor& self, StrideParams& strided_param) {
    strided_param.optype = kStridedOpSqueezeDims;
    strided_param.sizes = dims_vec;

    PT_VIEWTABLE_DEBUG(
        "squeeze dims fallback tensor id ",
        GetHbLazyTensorId(self),
        " dims ",
        dims_vec);
  };

  RUN_VIEW_OP_MAYBE_WITH_ACC_THREAD(squeeze_dims, self, out, param_setter);
}

Tensor& squeeze_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;

  return at::native::squeeze_(self);
}

Tensor& squeeze_dim_hpu_lazy_(Tensor& self, int64_t dim) {
  PT_LAZY_TRACE;

  return at::native::squeeze_(self, dim);
}

Tensor unsqueeze_hpu_lazy(const Tensor& self, int64_t dim_) {
  PT_LAZY_TRACE;

  auto dim = at::maybe_wrap_dim(dim_, self.dim() + 1);

  auto out = at::native::unsqueeze(self, dim);
  auto param_setter = [dim](const Tensor& self, StrideParams& strided_param) {
    strided_param.optype = kStridedOpUnsqueeze;
    StridedOpSqueezeParams squeeze_param = {dim};
    strided_param.params.squeeze_param = squeeze_param;

    PT_VIEWTABLE_DEBUG(
        "unsqueeze fallback tensor id ", GetHbLazyTensorId(self), " dim ", dim);
  };
  RUN_VIEW_OP_MAYBE_WITH_ACC_THREAD(unsqueeze, self, out, param_setter);
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
        : LazyOp<T>("hpu::permute_cl", vector_of_inputs, nullptr, -1) {}

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
    if (self.dim() == 0 && self.numel() == 1)
      return {self.clone()};
    auto out = at::native::permute(self, dims_in);
    habana::get_and_set_tensor_const(self, out);
    auto param_setter = [dims_in = dims_in.vec()](
                            const Tensor& self, StrideParams& strided_param) {
      strided_param.optype = kStridedOpPermute;
      strided_param.sizes = dims_in;
      PT_VIEWTABLE_DEBUG(
          "permute fallback tensor id ",
          GetHbLazyTensorId(self),
          " dims_in ",
          dims_in);
    };
    RUN_VIEW_OP_MAYBE_WITH_ACC_THREAD(permute, self, out, param_setter);
  } else {
    return permute_hpu_lazy_phy(self, dims_in);
  }
}

Tensor expand_hpu_lazy(const Tensor& self, SymIntArrayRef size, bool implicit) {
  PT_LAZY_TRACE;
  auto size_in = C10_AS_INTARRAYREF_SLOW(size);
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
    flush_op(1);
    return result;
  }

  auto out = at::native::expand(self, size_in, implicit);

  // lazy eager optimized view handling (no need to create view table)
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_VIEW_HANDLING) == true)) {
    return out;
  }

  auto additional_predicate = [](const Tensor& self, const Tensor& out) {
    auto self_id = GetHbLazyTensorId(self);
    auto out_id = GetHbLazyTensorId(out);
    return out_id != self_id;
  };

  auto param_setter = [size_in = size_in.vec(), implicit](
                          const Tensor& self, StrideParams& strided_param) {
    strided_param.optype = kStridedOpExpand;
    strided_param.sizes = size_in;
    StridedOpExpandParams expand_param = {implicit};
    strided_param.params.expand_param = expand_param;
    PT_VIEWTABLE_DEBUG(
        "expand fallback tensor id ",
        GetHbLazyTensorId(self),
        " sizes ",
        size_in);
  };

  RUN_WITH_PREDICATE_VIEW_OP_MAYBE_WITH_ACC_THREAD(
      expand, self, out, param_setter, additional_predicate);
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

  op_name = "hpu::topk";
  auto k_tensor = empty_hpu_lazy(
      k, self.options(), self.suggest_memory_format(), false, SHAPE_TENSOR);
  vector_of_inputs = {self, k_tensor, dim, largest, sorted};

  using T = std::tuple<at::Tensor, at::Tensor>;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(
        const Tensor& self,
        int64_t k,
        int64_t dim,
        const std::string& op_name,
        const std::vector<at::IValue>& vector_of_inputs)
        : LazyOp<T>(op_name, vector_of_inputs, nullptr, -1),
          self(self),
          k(k),
          dim(dim) {}

   private:
    T get_result_overrideable() override {
      auto shape_out = self.sizes().vec();
      int64_t dim_ = c10::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
      if (shape_out.size() > (uint64_t)(dim_)) {
        shape_out[dim_] = k;
      }
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

  Kernel kernel{self, k, dim, op_name, vector_of_inputs};
  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(topk, kernel)
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
    auto permuted_self = permute_hpu_lazy_phy(self, permute_dims);
    dim = self.dim() - 1;
    auto out =
        topk_hpu_lazy_impl(permuted_self, size_dim, dim, descending, true);
    auto permuted_out_0 = permute_hpu_lazy_phy(std::get<0>(out), permute_dims);
    auto permuted_out_1 = permute_hpu_lazy_phy(std::get<1>(out), permute_dims);
    return std::tie(permuted_out_0, permuted_out_1);
  } else
    return topk_hpu_lazy_impl(self, size_dim, dim, descending, true);
}

Scalar _local_scalar_dense_hpu(const Tensor& self) {
  PT_LAZY_TRACE;
  Scalar out;
  // If self is a lazy tensor make sure the execution till the point of self
  // getting flled has finished before we start copying
  if (IsHbLazyTensor(self)) {
    if (self.device().type() == c10::DeviceType::HPU) {
      flush_op();
      // Trigger point execution
      PT_IRGRAPH_DEBUG("step marker due to local scalar");
      HbLazyTensor::StepMarker({});
      StageSubmission::getInstance().setStageSubmissionFlow(
          StageSubmission::Mode::SET_WHEN_ANY_ITEM_CALL);
    }

    // if there is a view, we need to sync before accessing the tensor_data.
    // This is because we skip view outputs in stepmarker
    auto hb_tensor = GetHbLazyTensor(HbLazyTensorViews::HandleViewsD2H(self));
    auto tensor_data = hb_tensor.EvaluateTensorData();
    out = habana_helpers::_local_scalar_dense_internal(tensor_data);
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
  habana_lazy::NoAccThread no_acc_thread;

  auto result = empty_hpu_lazy(
      {1}, grad[0].options(), grad[0].suggest_memory_format(), false);

  handle_collective(grad);
  handle_collective(max_norm);

  auto op_func = [grad, max_norm, norm_type, result]() mutable {
    bool is_view_evaluated = true;

    for (size_t i = 0; i < grad.size(); i++) {
      auto& params_opt = GetHbLazyTensor(grad[i]).getDataPtr()->stride_params;

      if (params_opt.has_value()) {
        if (params_opt.value().viewStatus != kEvaluated) {
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
    node->set_as_output_tensor_list();
    habana_lazy::ir::Value& out1 = hlgrad.IrSetNode(node);

    habana_lazy::ir::NodePtr node_unpack =
        std::make_shared<habana_lazy::ir::ListUnpack>(out1);

    auto hlresult = GetHbLazyTensor(result);
    hlresult.IrSetNode(node_unpack, out_index++);

    // check if any of the grad is a view output and add strided insert node
    // accordingly
    for (size_t i = 0; i < grad.size(); i++) {
      auto grad_t = grad[i];
      auto hlgrad = GetHbLazyTensor(grad_t);
      auto& params_opt = hlgrad.getDataPtr()->stride_params;

      if ((!params_opt.has_value()) || (is_view_evaluated)) {
        hlgrad.IrSetNode(node_unpack, out_index++);
      } else {
        // fused norm has operated out of place on strided view's output
        auto clip_grad = empty_hpu_lazy(
            grad_t.sizes(),
            grad_t.options(),
            grad_t.suggest_memory_format(),
            false);
        auto hlgrad = GetHbLazyTensor(clip_grad);
        hlgrad.IrSetNode(node_unpack, out_index++);

        // add strided insert node. Do not flush in lazy eager as it is a
        // fused op. step marker will be used at the end
        strided_insert_hpu_lazy(grad_t, clip_grad, /*is_flush*/ false);
      }
    }

    flush_op(1);
  };
  // running hash
  {
    bool is_view_evaluated = true;

    for (size_t i = 0; i < grad.size(); i++) {
      auto hl_grad = GetHbLazyTensor(grad[i]);
      auto& params_opt = hl_grad.getDataPtr()->stride_params;

      if (params_opt.has_value()) {
        if (params_opt.value().viewStatus != kEvaluated) {
          is_view_evaluated = false;
          break;
        }
      } else {
        is_view_evaluated = false;
        break;
      }
    }
    std::vector<at::IValue> vector_of_inputs;
    for (auto t : grad) {
      vector_of_inputs.emplace_back(t);
    }
    vector_of_inputs.emplace_back(max_norm);
    vector_of_inputs.emplace_back(norm_type);
    if (is_view_evaluated) {
      RUNNING_HASH_COMBINE_OPERATOR(hpu::fused_norm_, vector_of_inputs);
    } else {
      RUNNING_HASH_COMBINE_OPERATOR(hpu::fused_norm_lazy, vector_of_inputs);
    }
  }
  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(fused_norm, op_func, result);
}

std::tuple<Tensor, Tensor> _unique_hpu_lazy(
    const Tensor& self,
    bool sorted,
    bool return_inverse) {
  PT_LAZY_TRACE;

  habana_lazy::NoAccThread no_acc_thread;

  if (self.numel() == 0) {
    auto shape = DimVector{0};
    auto output = empty_hpu_lazy(
        shape, self.options(), self.suggest_memory_format(), true);
    auto inverse_tensor = empty_hpu_lazy(
        shape,
        self.options().dtype(c10::ScalarType::Long),
        self.suggest_memory_format(),
        true);
    auto hl_inverse_tensor = GetHbLazyTensor(inverse_tensor);
    flush_op(2);
    return {output, inverse_tensor};
  }

  struct Unique : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> {
    explicit Unique(
        const std::vector<at::IValue>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>>(
              "hpu::_unique",
              inputs,
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
      {self, sorted, return_inverse},
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
  flush_op(1);

  if (return_inverse) {
    auto inverse_tensor = std::get<2>(output);
    // Index flipping to match the cpu results
    Tensor subtracter = add_scalar_hpu_lazy(valid_count, 1, -1);
    auto inverse_result = add_tensor_hpu_lazy(subtracter, inverse_tensor, -1);
    inverse_result =
        view_hpu(inverse_result, fromIntArrayRefUnchecked(self.sizes()));

    flush_op(1);
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

  habana_lazy::NoAccThread no_acc_thread;

  if (self.numel() == 0) {
    auto result_ = empty_hpu_lazy(
        self.sizes(), self.options(), self.suggest_memory_format(), true);
    return std::make_tuple(result_, result_, result_);
  }
  struct Unique
      : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>> {
    explicit Unique(
        const std::vector<at::IValue>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>(
              "hpu::_unique2",
              inputs,
              out_shapes,
              -1) {}

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    get_result_overrideable() override {
      auto inputs = get_inputs();
      auto self = inputs[0].toTensor();
      int elements = self.numel();
      auto output_shape = at::DimVector{elements};
      auto valid_shape = at::DimVector{1};
      auto inverse_tensor_shape = DimVector{elements};
      auto counts_tensor_shape = DimVector{elements};
      auto result0 = empty_hpu_lazy(
          output_shape, self.options(), self.suggest_memory_format(), false);
      auto result1 = empty_hpu_lazy(
          valid_shape,
          self.options().dtype(c10::ScalarType::Int),
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

  int elements = self.numel();
  std::vector<int64_t> feature_map_shape{elements};
  std::vector<int64_t> valid_count_shape{1};
  std::vector<int64_t> inverse_tensor_shape{elements};
  std::vector<int64_t> counts_tensor_shape{elements};
  // Add unique_2 node
  Unique k(
      {self, sorted, return_inverse, return_counts},
      {feature_map_shape,
       valid_count_shape,
       inverse_tensor_shape,
       counts_tensor_shape});
  // unique2 returns 2 output feature_map and valid tensor
  auto output = k.call();
  auto feature_map = std::get<0>(output);
  auto valid_count = std::get<1>(output);
  auto inverse_tensor = std::get<2>(output);
  inverse_tensor = torch::reshape(inverse_tensor, self.sizes());
  auto counts_tensor = std::get<3>(output);

  // Force an execution here because "unique" is a non shape inferable op.
  // .item() internally triggers a mark_step
  PT_IRGRAPH_DEBUG("step marker due to unique");
  auto end = valid_count.item<int64_t>();
  StageSubmission::getInstance().setStageSubmissionFlow();

  // Add a slice node to capture relevent elements from feature_map
  auto unique_result = slice_hpu_lazy(feature_map, 0, 0, end, 1);
  auto counts_result = slice_hpu_lazy(counts_tensor, 0, 0, end, 1);

  flush_op(3);

  // These are optional tensors which shall be populated only when we
  // start supporting return_inverse and return_counts

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

std::tuple<Tensor, Tensor, Tensor> unique_dim_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  PT_LAZY_TRACE;

  habana_lazy::NoAccThread no_acc_thread;

  if (dim < 0) {
    dim = self.dim() + dim;
  }
  if (self.numel() == 0) {
    auto shape = DimVector{0};
    auto output = empty_hpu_lazy(
        shape, self.options(), self.suggest_memory_format(), true);
    flush_op(1);
    auto inverse_tensor = empty_hpu_lazy(
        shape,
        self.options().dtype(c10::ScalarType::Long),
        self.suggest_memory_format(),
        true);
    auto hl_inverse_tensor = GetHbLazyTensor(inverse_tensor);
    flush_op(1);
    auto counts_tensor = empty_hpu_lazy(
        shape,
        self.options().dtype(c10::ScalarType::Long),
        self.suggest_memory_format(),
        true);
    auto hl_counts_tensor = GetHbLazyTensor(counts_tensor);
    flush_op(1);
    return {output, inverse_tensor, counts_tensor};
  }

  struct Unique
      : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>> {
    explicit Unique(
        const std::vector<at::IValue>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes = {})
        : LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>(
              "hpu::unique_dim",
              inputs,
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
      {self, dim, sorted, return_inverse, return_counts},
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

  flush_op(3);

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

Tensor matmul_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    c10::optional<at::ScalarType> dtype) {
  PT_LAZY_TRACE;
  c10::ScalarType out_dtype =
      dtype.has_value() ? dtype.value() : self.dtype().toScalarType();
  LazyOp<Tensor> k(
      "aten::matmul",
      {self, other},
      {MatMulOperator::compute_output_shape(self, other)});
  k.set_scalar_types({out_dtype});
  RUN_MAYBE_WITH_ACC_THREAD(matmul, k)
}

std::tuple<Tensor, Tensor> matmul_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& other,
    c10::optional<at::ScalarType> dtype) {
  PT_LAZY_TRACE;
  c10::ScalarType out_dtype =
      dtype.has_value() ? dtype.value() : self.dtype().toScalarType();
  LazyOp<std::tuple<Tensor, Tensor>> k(
      "hpu::matmul_backward",
      {grad_output, self, other},
      {self.sizes().vec(), other.sizes().vec()});
  k.set_scalar_types({out_dtype, out_dtype});
  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(matmul_backward, k)
}

Tensor habana_nms_hpu_lazy(
    const Tensor& boxes,
    const Tensor& scores,
    float iou_threshold) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_NMS_USING_BNMS_CGUID)) {
    auto options = scores.options();
    auto indices_tensor = empty_hpu_lazy(
        scores.sizes(),
        options.dtype(torch::kInt32),
        scores.suggest_memory_format(),
        true);
    fill_hpu_lazy_(indices_tensor, 0);
    return batched_nms_hpu_lazy(boxes, scores, indices_tensor, iou_threshold);
  }

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
        "hpu::cast", {boxes, c10::ScalarType::Float}, {boxes.sizes().vec()}};
    k_.set_scalar_types({c10::ScalarType::Float});
    boxes_cast = k_.call();
  }
  if (scores.scalar_type() == c10::ScalarType::BFloat16) {
    LazyOp<at::Tensor> s_{
        "hpu::cast", {scores, c10::ScalarType::Float}, {scores.sizes().vec()}};
    s_.set_scalar_types({c10::ScalarType::Float});
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
  float score_threshold{-std::numeric_limits<float>::max()};
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
  flush_op(1);
  return result;
}

Tensor batched_nms_hpu_lazy(
    const Tensor& boxes,
    const Tensor& scores,
    const Tensor& indexes,
    float iou_threshold) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  if (boxes.numel() == 0 && scores.numel() == 0 && indexes.numel() == 0) {
    auto shape = DimVector{0};
    auto output = empty_hpu_lazy(
        shape,
        scores.options().dtype(c10::ScalarType::Long),
        scores.suggest_memory_format(),
        true);
    auto hl_output = GetHbLazyTensor(output);
    flush_op(1);
    return output;
  }
  // Ensuring that the boxes and scores input to batched_nms is always FP32,
  // this is because CGUID expects boxes to be f32. TBD: move this cast
  // addition to HMP
  Tensor boxes_cast = boxes;
  Tensor scores_cast = scores;
  if (boxes.scalar_type() == c10::ScalarType::BFloat16) {
    LazyOp<at::Tensor> k_{
        "hpu::cast", {boxes, c10::ScalarType::Float}, {boxes.sizes().vec()}};
    k_.set_scalar_types({c10::ScalarType::Float});
    boxes_cast = k_.call();
  }
  if (scores.scalar_type() == c10::ScalarType::BFloat16) {
    LazyOp<at::Tensor> s_{
        "hpu::cast", {scores, c10::ScalarType::Float}, {scores.sizes().vec()}};
    s_.set_scalar_types({c10::ScalarType::Float});
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
  StageSubmission::getInstance().setStageSubmissionFlow();

  // Extract correct output using shape information.
  // Add a slice node to capture relevent elements
  auto result = slice_hpu_lazy(box_id_out, 0, 0, end, 1);
  flush_op(1);
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
  PT_LAZY_OP_TRACE;
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
        "hpu::cast", {rois, c10::ScalarType::Float}, {rois.sizes().vec()}});
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
  PT_LAZY_OP_TRACE;
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
      {out_shape});
  RUN_MAYBE_WITH_ACC_THREAD(roi_align_bwd, k)
}

at::Tensor& broadcast_hpu_lazy_(
    at::Tensor& tensor,
    int64_t root_rank,
    int64_t comm_id) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;
  HbLazyTensorViews::HandleViewsLazyCollective(tensor);
  MarkTensorAsOutputFromCollectiveOp(tensor);

  LazyOp<at::Tensor&> k("hccl::broadcast_", {tensor, root_rank, comm_id});
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(broadcast_, k, tensor)
}

at::Tensor& allreduce_hpu_lazy_(
    at::Tensor& tensor,
    uint8_t reduce_op,
    int64_t comm_id) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;
  HbLazyTensorViews::HandleViewsLazyCollective(tensor);
  MarkTensorAsOutputFromCollectiveOp(tensor);
  LazyOp<at::Tensor&> k("hccl::allreduce_", {tensor, reduce_op, comm_id});
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(allreduce_, k, tensor)
}

at::Tensor& reduce_hpu_lazy_(
    at::Tensor& tensor,
    int64_t dst_rank,
    uint8_t reduce_op,
    int64_t comm_id) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;
  HbLazyTensorViews::HandleViewsLazyCollective(tensor);
  MarkTensorAsOutputFromCollectiveOp(tensor);
  LazyOp<at::Tensor&> k(
      "hccl::reduce_", {tensor, dst_rank, reduce_op, comm_id});
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(reduce_, k, tensor)
}

at::Tensor& alltoall_hpu_lazy_out(
    const at::Tensor& inputTensor,
    int64_t comm_id,
    at::Tensor& outputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;
  HbLazyTensorViews::HandleViewsLazyCollective(inputTensor);
  MarkTensorAsOutputFromCollectiveOp(outputTensor);
  LazyOp<at::Tensor&> k(
      "hccl::alltoall_out",
      {inputTensor, comm_id, outputSplitSizes, inputSplitSizes, outputTensor});
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(alltoall_out, k, outputTensor)
}

at::Tensor& allgather_hpu_lazy_out(
    const at::Tensor& inputTensor,
    int64_t comm_id,
    at::Tensor& outputTensor) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;
  HbLazyTensorViews::HandleViewsLazyCollective(inputTensor);
  MarkTensorAsOutputFromCollectiveOp(outputTensor);
  LazyOp<at::Tensor&> k(
      "hccl::allgather_out",
      {inputTensor, comm_id, outputTensor},
      {outputTensor.sizes().vec()});
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(allgather_out, k, outputTensor)
}

at::Tensor& reduce_scatter_hpu_lazy_out(
    const at::Tensor& inputTensor,
    uint8_t reduce_op,
    int64_t comm_id,
    at::Tensor& outputTensor) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;
  MarkTensorAsOutputFromCollectiveOp(outputTensor);
  LazyOp<at::Tensor&> k(
      "hccl::reduce_scatter_out",
      {inputTensor, reduce_op, comm_id, outputTensor},
      {outputTensor.sizes().vec()});
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(reduce_scatter_out, k, outputTensor)
}

at::Tensor& send_hpu_lazy_(
    at::Tensor& tensor,
    int64_t dst_rank,
    int64_t tag,
    int64_t comm_id) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;
  HbLazyTensorViews::HandleViewsLazyCollective(tensor);
  MarkTensorAsOutputFromCollectiveOp(tensor);
  LazyOp<at::Tensor&> k("hccl::send_", {tensor, dst_rank, tag, comm_id});
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(send_, k, tensor)
}

at::Tensor& recv_hpu_lazy_(
    at::Tensor& tensor,
    int64_t src_rank,
    int64_t tag,
    int64_t comm_id) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;
  HbLazyTensorViews::HandleViewsLazyCollective(tensor);
  MarkTensorAsOutputFromCollectiveOp(tensor);
  LazyOp<at::Tensor&> k("hccl::recv_", {tensor, src_rank, tag, comm_id});
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(recv_, k, tensor)
}

std::tuple<at::Tensor&, at::Tensor&> cast_to_fp8_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  LazyOp<std::tuple<at::Tensor&, at::Tensor&>> hpu_op{
      "hpu::cast_to_fp8",
      {input, scale, stochastic_rounding, out, amax},
      {input.sizes().vec(), amax.sizes().vec()}};
  auto result = ::std::tuple<at::Tensor&, at::Tensor&>(out, amax);

  RUN_INPLACE_TUPLE_MAYBE_WITH_ACC_THREAD(cast_to_fp8, hpu_op, result)
}

template <class T>
static std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_common(
    const at::Tensor& input,
    T scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef scale_shape) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "cast_to_fp8_v2:",
      DUMP_6ARGS(
          input, scale, stochastic_rounding, is_amax, dtype, scale_shape));
  FP8_CHECK
  LazyOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::cast_to_fp8_v2",
      {input, scale, stochastic_rounding, is_amax, dtype, scale_shape},
      CastToFp8V2OutputShape};
  hpu_op.set_scalar_types({dtype, at::ScalarType::Float});

  RUN_MAYBE_WITH_ACC_THREAD(cast_to_fp8_v2, hpu_op)
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef scale_shape) {
  return cast_to_fp8_v2_common(
      input, scale, stochastic_rounding, is_amax, dtype, scale_shape);
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_scalar_lazy(
    const at::Tensor& input,
    double scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef) {
  return cast_to_fp8_v2_common(
      input, scale, stochastic_rounding, is_amax, dtype, c10::nullopt);
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_scalar_list_lazy(
    const at::Tensor& input,
    c10::ArrayRef<double> scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef scale_shape) {
  return cast_to_fp8_v2_common(
      input, scale, stochastic_rounding, is_amax, dtype, scale_shape);
}

template <class T>
static at::Tensor cast_from_fp8_common(
    const at::Tensor& input,
    T scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef scale_shape) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "cast_from_fp8:", DUMP_4ARGS(input, scale, out_dtype, scale_shape));
  FP8_CHECK
  LazyOp<at::Tensor> hpu_op{
      "hpu::cast_from_fp8",
      {input, scale, out_dtype, scale_shape},
      {input.sizes().vec()}};
  hpu_op.set_scalar_types({out_dtype});
  RUN_MAYBE_WITH_ACC_THREAD(cast_from_fp8, hpu_op)
}

at::Tensor cast_from_fp8_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef scale_shape) {
  return cast_from_fp8_common(input, scale, out_dtype, scale_shape);
}

at::Tensor cast_from_fp8_scalar_lazy(
    const at::Tensor& input,
    double scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef) {
  return cast_from_fp8_common(input, scale, out_dtype, c10::nullopt);
}

at::Tensor cast_from_fp8_scalar_list_lazy(
    const at::Tensor& input,
    c10::ArrayRef<double> scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef scale_shape) {
  return cast_from_fp8_common(input, scale, out_dtype, scale_shape);
}

at::Tensor convert_from_int4_common(
    const std::string& op_name,
    const at::Tensor& input,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& zero_point,
    at::ScalarType out_dtype) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(op_name + " :", DUMP_4ARGS(input, scale, zero_point, out_dtype));

  auto output_shape = input.sizes().vec();
  output_shape.back() *= 8;

  LazyOp<at::Tensor> hpu_op{
      "hpu::" + op_name, {input, scale, zero_point, out_dtype}, {output_shape}};
  hpu_op.set_scalar_types({out_dtype});
  RUN_MAYBE_WITH_ACC_THREAD(convert_from_int4, hpu_op)
}

at::Tensor convert_from_int4_lazy(
    const at::Tensor& input,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& zero_point,
    at::ScalarType out_dtype) {
  return convert_from_int4_common(
      "convert_from_int4", input, scale, zero_point, out_dtype);
}

at::Tensor convert_from_uint4_lazy(
    const at::Tensor& input,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& zero_point,
    at::ScalarType out_dtype) {
  return convert_from_int4_common(
      "convert_from_uint4", input, scale, zero_point, out_dtype);
}

at::Tensor& fp8_gemm_lazy(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const at::Tensor& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& A_scale_inv,
    const c10::optional<at::Tensor>& B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    at::Tensor& out) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  int64_t rank = A.dim();
  std::vector<int64_t> A_shape = A.sizes().vec();
  std::vector<int64_t> B_shape = B.sizes().vec();
  std::vector<int64_t> out_shape{A_shape.begin(), A_shape.begin() + rank - 2};
  int A_dim = rank - 2 + (trans_A ? 1 : 0);
  int B_dim = rank - 2 + (trans_B ? 0 : 1);
  out_shape.push_back(A_shape[A_dim]);
  out_shape.push_back(B_shape[B_dim]);

  LazyOp<at::Tensor&> hpu_op{
      "hpu::fp8_gemm",
      {A,
       trans_A,
       B,
       trans_B,
       D,
       out_dtype,
       A_scale_inv,
       B_scale_inv,
       bias,
       accumulate,
       out},
      {out_shape}};
  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(fp8_gemm, hpu_op, out)
}

template <class T>
static at::Tensor fp8_gemm_v2_common(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    T A_scale_inv,
    T B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> hpu_op{
      "hpu::fp8_gemm_v2",
      {A,
       trans_A,
       B,
       trans_B,
       D,
       out_dtype,
       A_scale_inv,
       B_scale_inv,
       bias,
       accumulate,
       B_scale_shape},
      Fp8GemmV2OutputShape};
  hpu_op.set_scalar_types({out_dtype});
  RUN_MAYBE_WITH_ACC_THREAD(fp8_gemm_v2, hpu_op)
}

at::Tensor fp8_gemm_v2_lazy(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& A_scale_inv,
    const c10::optional<at::Tensor>& B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape) {
  return fp8_gemm_v2_common(
      A,
      trans_A,
      B,
      trans_B,
      D,
      out_dtype,
      A_scale_inv,
      B_scale_inv,
      bias,
      accumulate,
      B_scale_shape);
}

at::Tensor fp8_gemm_v2_lazy_scalar(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    double A_scale_inv,
    double B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef) {
  return fp8_gemm_v2_common(
      A,
      trans_A,
      B,
      trans_B,
      D,
      out_dtype,
      A_scale_inv,
      B_scale_inv,
      bias,
      accumulate,
      c10::nullopt);
}

at::Tensor fp8_gemm_v2_lazy_scalar_list(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    c10::ArrayRef<double> A_scale_inv,
    c10::ArrayRef<double> B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape) {
  return fp8_gemm_v2_common(
      A,
      trans_A,
      B,
      trans_B,
      D,
      out_dtype,
      A_scale_inv,
      B_scale_inv,
      bias,
      accumulate,
      B_scale_shape);
}

inline bool is_main_thread_and_lazy_collectives_enabled() {
  return GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_COLLECTIVES) &&
      not(habana_lazy::AccThread::IsAccThreadEnabled() &&
          habana_lazy::AccThread::Get().inAccThreadContext());
}

inline bool is_hpu_tensor(const at::Tensor& tensor) {
  return tensor.defined() && tensor.device().type() == c10::DeviceType::HPU;
}

void handle_collective(const at::IValue& value) {
  // call GetHbLazyTensor to trigger StepMarker in the main thread for
  // outputs from lazy collective operations
  if (value.isTensor()) {
    handle_collective(value.toTensor());
  } else if (value.isTensorList()) {
    handle_collective(value.toTensorVector());
  }
  // else do nothing
}

void handle_collective(const at::Tensor& tensor) {
  if (!is_main_thread_and_lazy_collectives_enabled())
    return;

  if (!is_hpu_tensor(tensor))
    return;

  auto hl_t = GetHbLazyTensor(tensor);
  // also check for tensor's parent in case it's a view tensor. GetHbLazyTensor
  // will trigger markstep if parent tensor is produced from a collective op.
  // This markstep should be triggered before handleViews, otherwise in between
  // op processing markstep will be triggered which can cause some side effects.
  while (hl_t.getDataPtr()->stride_params.has_value()) {
    auto& params = hl_t.getDataPtr()->stride_params.value();
    auto parent_or_base =
        (params.optype == kStridedOpDefault) ? params.base : params.parent;
    hl_t = GetHbLazyTensor(
        HbLazyTensorViews::get_recent_base_tensor(parent_or_base));
  }
}

template <typename It, typename Sentinel>
void handle_collective(It iter, Sentinel end) {
  if (!is_main_thread_and_lazy_collectives_enabled())
    return;

  for (; iter != end; ++iter) {
    if (!is_hpu_tensor(*iter))
      continue;

    GetHbLazyTensor(*iter);
  }
}

void handle_collective(const at::TensorList& list) {
  handle_collective(std::begin(list), std::end(list));
}

void handle_collective(const std::vector<at::Tensor>& vec) {
  handle_collective(std::begin(vec), std::end(vec));
}

void handle_collective(const at::ITensorListRef& list) {
  handle_collective(std::begin(list), std::end(list));
}

at::Tensor habana_random_seed_lazy(const at::Tensor& input) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  std::vector<int64_t> out_shape = input.sizes().vec();
  struct Kernel : public LazyOp<at::Tensor> {
    explicit Kernel(const Tensor& input, std::vector<int64_t> out_shape)
        : LazyOp<at::Tensor>("hpu::habana_random_seed", {input}, {out_shape}) {}
  };
  Kernel kernel{input, std::move(out_shape)};
  return kernel.call();
}

at::Tensor _copy_from(const at::Tensor&, const at::Tensor&, bool) {
  HABANA_ASSERT(
      false,
      "This function should not be called in lazy flow. Something went wrong.");
  std::terminate();
}

at::Tensor& set_source_Storage(at::Tensor&, at::Storage) {
  HABANA_ASSERT(
      false,
      "This function should not be called in lazy flow. Something went wrong.");
  std::terminate();
}

at::Tensor& set_source_Tensor(at::Tensor&, const at::Tensor&) {
  HABANA_ASSERT(
      false,
      "This function should not be called in lazy flow. Something went wrong.");
  std::terminate();
}

at::Tensor& set_(at::Tensor&) {
  HABANA_ASSERT(
      false,
      "This function should not be called in lazy flow. Something went wrong.");
  std::terminate();
}

template <
    typename Tout,
    bool is1D,
    bool hasWeights = (std::tuple_size_v<Tout> == 3)>
std::vector<at::Tensor> habana_permute_1D_2D_sparse_data_helper_lazy(
    const std::vector<at::IValue>& inputs) {
  class Kernel : public LazyOp<Tout> {
   public:
    Kernel(const std::string& op_name, const std::vector<at::IValue>& inputs)
        : LazyOp<Tout>(
              op_name,
              inputs,
              std::vector<std::vector<int64_t>>(),
              -1) {}

   private:
    std::vector<at::IValue>& get_inputs() {
      return LazyOp<Tout>::get_inputs();
    }

    Tout get_result_overrideable() override {
      const auto& inputs = get_inputs();
      auto lengths = inputs[1].toTensor();
      auto indices = inputs[2].toTensor();

      auto create_res = [&](Tensor in) {
        // Applies to lenghts only
        if (is1D && in.dim() == 2) {
          auto shape = in.sizes().vec();
          auto shape_new = {shape[0] * shape[1]};
          return empty_hpu_lazy(
              shape_new, in.options(), in.suggest_memory_format(), false);
        }

        return empty_hpu_lazy(
            in.sizes(), in.options(), in.suggest_memory_format(), false);
      };

      if constexpr (hasWeights)
        return {
            create_res(lengths),
            create_res(indices),
            create_res(inputs[3].toTensor())};
      else
        return {create_res(lengths), create_res(indices)};
    };
  };

  std::string hpu_op_name = "hpu::habana_permute_";
  hpu_op_name += is1D ? "1D" : "2D";
  hpu_op_name += hasWeights ? "_sparse_data" : "_sparse_data_without_weights";

  Kernel k{hpu_op_name, inputs};

  std::vector<at::Tensor> out_v;
  auto out = k.get_result();
  for_each_in_tuple(
      out, [&out_v](const auto& result) { out_v.push_back(result); });
  auto func = [op = std::move(k), out_v = std::move(out_v)]() mutable {
    if constexpr (hasWeights)
      op.call(std::tie(out_v[0], out_v[1], out_v[2]));
    else
      op.call(std::tie(out_v[0], out_v[1]));
  };

  std::vector<at::Tensor> res_vec{std::get<0>(out), std::get<1>(out)};
  if constexpr (hasWeights)
    res_vec.emplace_back(std::get<02>(out));

  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD(
      habana_permute_1D_sparse_data, func, res_vec)
}

std::vector<at::Tensor> habana_permute_1D_sparse_data_lazy(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  std::vector<at::IValue> inputs = {permute, lengths, indices};

  if (weights) {
    inputs.push_back(weights);
    return habana_permute_1D_2D_sparse_data_helper_lazy<
        std::tuple<Tensor, Tensor, Tensor>,
        true>(inputs);
  } else {
    return habana_permute_1D_2D_sparse_data_helper_lazy<
        std::tuple<Tensor, Tensor>,
        true>(inputs);
  }
}

std::vector<at::Tensor> habana_permute_2D_sparse_data_lazy(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  std::vector<at::IValue> inputs = {permute, lengths, indices};

  if (weights) {
    inputs.push_back(weights);
    return habana_permute_1D_2D_sparse_data_helper_lazy<
        std::tuple<Tensor, Tensor, Tensor>,
        false>(inputs);
  } else {
    return habana_permute_1D_2D_sparse_data_helper_lazy<
        std::tuple<Tensor, Tensor>,
        false>(inputs);
  }
}

at::Tensor habana_expand_into_jagged_permute_lazy(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t output_size) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op{
      "hpu::habana_expand_into_jagged_permute",
      {permute, input_offsets, output_offsets, output_size},
      {{output_size}}};
  RUN_MAYBE_WITH_ACC_THREAD(expand_into_jagged_permute, op)
}

at::Tensor mixture_of_experts_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}
at::Tensor mixture_of_experts_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor habana_split_permute_cat_lazy(
    const at::Tensor& input,
    const at::Tensor& indices,
    int64_t batch_size,
    int64_t num_features,
    int64_t dims) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op{
      "hpu::habana_split_permute_cat",
      {input, indices, batch_size, num_features, dims},
      {{input.sizes().vec()}}};

  RUN_MAYBE_WITH_ACC_THREAD(split_permute_cat, op)
}

at::Tensor _ragged_softmax(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float,
    const at::Tensor& valid_count) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "HpuOp _ragged_softmax :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " half_to_float=",
      to_string(half_to_float));

  LazyOp<at::Tensor> hpu_op{
      "hpu::ragged_softmax", {self, dim, half_to_float, valid_count}};
  RUN_MAYBE_WITH_ACC_THREAD(_ragged_softmax, hpu_op);
}

at::Tensor scaled_masked_softmax_lazy(
    const at::Tensor& input,
    const at::Tensor& mask,
    double scale) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op{
      "hpu::scaled_masked_softmax",
      {input, mask, scale},
      {{input.sizes().vec()}}};
  RUN_MAYBE_WITH_ACC_THREAD(scaled_masked_softmax, op)
}

at::Tensor custom_softmax_lazy(const at::Tensor& input, int64_t flavor) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op{
      "hpu::custom_softmax", {input, flavor}, {{input.sizes().vec()}}};
  RUN_MAYBE_WITH_ACC_THREAD(custom_softmax, op)
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
habana_bounds_check_indices_lazy(
    at::Tensor& indices,
    at::Tensor& offsets,
    at::Tensor& warning,
    const at::Tensor& rows_per_table,
    int64_t bounds_check_mode,
    const c10::optional<at::Tensor>& weights) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>> op{
      "hpu::habana_bounds_check_indices",
      {indices, offsets, warning, rows_per_table, bounds_check_mode, weights}};
  auto result = ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(
      indices, offsets, warning);

  RUN_INPLACE_TUPLE_MAYBE_WITH_ACC_THREAD(bounds_check_indices, op, result)
}

at::Tensor optimizer_lamb_norm_hpu_lazy(
    const std::vector<at::Tensor>& grad,
    double max_grad_norm) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      " optimizer_lamb_norm:",
      " grad=",
      to_string(grad),
      "max_grad_norm=",
      to_string(max_grad_norm));

  LazyOptimizerLambNorm<at::Tensor> op{
      "hpu::optimizer_lamb_fused_norm", {grad, max_grad_norm}};

  RUN_MAYBE_WITH_ACC_THREAD(optimizer_lamb_norm, op)
}

void optimizer_lamb_phase1(
    const at::TensorList gradients,
    const at::TensorList weights,
    at::TensorList exp_avg,
    at::TensorList exp_avg_sq,
    at::TensorList out_weight_norms,
    at::TensorList out_adam_norms,
    at::TensorList out_adam_steps,
    const at::Tensor& clip_global_grad_norm,
    const int64_t grad_averaging,
    const double beta1,
    const double beta2,
    const double epsilon,
    const at::Tensor& bias_correction1,
    const at::Tensor& bias_correction2,
    const double weight_decay) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "optimizer_lamb_phase1:",
      DUMP_12ARGS(
          gradients,
          weights,
          exp_avg,
          exp_avg_sq,
          clip_global_grad_norm,
          grad_averaging,
          beta1,
          beta2,
          epsilon,
          bias_correction1,
          bias_correction2,
          weight_decay));

  LazyOp<void> hpu_op{
      "hpu::optimizer_lamb_phase1",
      {gradients,
       weights,
       exp_avg,
       exp_avg_sq,
       out_weight_norms,
       out_adam_norms,
       out_adam_steps,
       clip_global_grad_norm,
       grad_averaging,
       beta1,
       beta2,
       epsilon,
       bias_correction1,
       bias_correction2,
       weight_decay},
      std::vector<std::vector<int64_t>>{},
      -1};

  return hpu_op.call(std::vector<at::TensorList>{
      exp_avg, exp_avg_sq, out_weight_norms, out_adam_norms, out_adam_steps});
}

void optimizer_lamb_phase2(
    at::TensorList weights,
    const at::TensorList adam_norms,
    const at::TensorList weight_norms,
    const at::TensorList adam_steps,
    const at::Tensor& neg_step,
    const double weight_decay,
    const bool use_lamb) {
  PT_LAZY_OP_TRACE
  PT_LAZY_TRACE;

  PT_OP_INFO(
      "optimizer_lamb_phase2 :",
      DUMP_7ARGS(
          weights,
          adam_norms,
          weight_norms,
          adam_steps,
          neg_step,
          weight_decay,
          use_lamb));

  LazyOptimizationOp<void> loo(
      "hpu::optimizer_lamb_phase2",
      {weights,
       adam_norms,
       weight_norms,
       adam_steps,
       neg_step,
       weight_decay,
       use_lamb});
  loo.call(weights);
  flush_op(weights.size());
}

at::Tensor rotary_pos_embedding_lazy(
    const at::Tensor& input,
    const at::Tensor& sin,
    const at::Tensor& cos,
    const c10::optional<at::Tensor>& position_ids,
    const int64_t offset,
    const int64_t mode) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op{
      "hpu::rotary_pos_embedding",
      {input, sin, cos, position_ids, offset, mode},
      {{input.sizes().vec()}}};

  RUN_MAYBE_WITH_ACC_THREAD(rotary_pos_embedding, op)
}

at::Tensor rotary_pos_embedding_backward_lazy(
    const at::Tensor& grad_in,
    const at::Tensor& sin,
    const at::Tensor& cos,
    const c10::optional<at::Tensor>& position_ids,
    const int64_t offset,
    const int64_t mode) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op{
      "hpu::rotary_pos_embedding_backward",
      {grad_in, sin, cos, position_ids, offset, mode},
      {{grad_in.sizes().vec()}}};

  RUN_MAYBE_WITH_ACC_THREAD(rotary_pos_embedding_backward, op)
}

std::tuple<at::Tensor, at::Tensor> ctc_loss_custom_lazy(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  auto shapes = habana::calculate_output_shapes_for_ctc_loss_custom_fwd(
      log_probs, targets, reduction);

  LazyOp<std::tuple<at::Tensor, at::Tensor>> op{
      "hpu::ctc_loss_custom",
      {log_probs,
       targets,
       input_lengths,
       target_lengths,
       blank,
       reduction,
       zero_infinity},
      {std::get<0>(shapes), std::get<1>(shapes)}};

  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(ctc_loss_custom, op)
}

at::Tensor ctc_loss_custom_backward_lazy(
    const at::Tensor& grad,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op{
      "hpu::ctc_loss_custom_backward",
      {grad,
       log_probs,
       targets,
       input_lengths,
       target_lengths,
       neg_log_likelihood,
       log_alpha,
       blank,
       reduction,
       zero_infinity},
      {{log_probs.sizes().vec()}}};

  RUN_MAYBE_WITH_ACC_THREAD(ctc_loss_custom_backward, op)
}

at::Tensor masked_batch_gemm_lazy(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& mask_a,
    const at::Tensor& mask_b,
    bool trans_a,
    bool trans_b) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> hpu_op{
      "hpu::masked_batch_gemm",
      {a, b, mask_a, mask_b, trans_a, trans_b},
      MaskedBatchGemmOutputShape};

  RUN_MAYBE_WITH_ACC_THREAD(masked_batch_gemm, hpu_op)
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd_lazy(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  if (p > 0.0) {
    c10::optional<Generator> gen;
    auto seed = habana::get_seed_tensor_hpu(gen);
    LazyOp<std::tuple<Tensor, Tensor, Tensor>> hpu_op{
        "hpu::sdpa_fwd_dropout_seed",
        {seed,
         q,
         k,
         v,
         attention_mask,
         p,
         scale,
         is_causal,
         softmax_mode,
         valid_seq_len,
         seq_padding_type},
        SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {q.scalar_type(), q.scalar_type(), c10::ScalarType::Char});
    RUN_TUPLE_MAYBE_WITH_ACC_THREAD(sdpa_recomp_fwd, hpu_op)
  } else {
    LazyOp<std::tuple<Tensor, Tensor, Tensor>> hpu_op{
        "hpu::sdpa_fwd",
        {q,
         k,
         v,
         attention_mask,
         p,
         scale,
         is_causal,
         softmax_mode,
         valid_seq_len,
         seq_padding_type},
        SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {q.scalar_type(), q.scalar_type(), c10::ScalarType::Char});
    RUN_TUPLE_MAYBE_WITH_ACC_THREAD(sdpa_recomp_fwd, hpu_op)
  }
}

template <class T>
static std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_fwd_common(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    T d_scale_q,
    T d_scale_k,
    T d_scale_v,
    T q_scale_s,
    T q_scale_o,
    T d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type,
    c10::ScalarType fwdOutType) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "fp8_sdpa_recomp_fwd :",
      DUMP_19ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          requires_backward,
          softmax_mode,
          d_scale_q,
          d_scale_k,
          d_scale_v,
          q_scale_s,
          q_scale_o,
          d_scale_s,
          is_amax_s,
          is_amax_o,
          valid_seq_len,
          seq_padding_type));

  auto linvType = c10::ScalarType::Float;

  if ((softmax_mode == "fast") &&
      (q.scalar_type() == c10::ScalarType::BFloat16)) {
    linvType = c10::ScalarType::BFloat16;
  }
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    linvType = c10::ScalarType::BFloat16;
  }

  auto mType = q.scalar_type();
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    mType = c10::ScalarType::BFloat16;
  }

  if (p > 0.0) {
    c10::optional<Generator> gen;
    auto seed = habana::get_seed_tensor_hpu(gen);
    LazyOp<std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> hpu_op{
        "hpu::fp8_sdpa_recomp_fwd_dropout_seed",
        {seed,
         q,
         k,
         v,
         attention_mask,
         p,
         scale,
         is_causal,
         requires_backward,
         softmax_mode,
         d_scale_q,
         d_scale_k,
         d_scale_v,
         q_scale_s,
         q_scale_o,
         d_scale_s,
         is_amax_s,
         is_amax_o,
         valid_seq_len,
         seq_padding_type},
        Fp8SDPARecompFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType,
         mType,
         linvType,
         c10::ScalarType::Int,
         c10::ScalarType::Float,
         c10::ScalarType::Float});

    RUN_TUPLE_MAYBE_WITH_ACC_THREAD(fp8_sdpa_recomp_fwd, hpu_op)
  } else {
    LazyOp<std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> hpu_op{
        "hpu::fp8_sdpa_recomp_fwd",
        {q,
         k,
         v,
         attention_mask,
         p,
         scale,
         is_causal,
         requires_backward,
         softmax_mode,
         d_scale_q,
         d_scale_k,
         d_scale_v,
         q_scale_s,
         q_scale_o,
         d_scale_s,
         is_amax_s,
         is_amax_o,
         valid_seq_len,
         seq_padding_type},
        Fp8SDPARecompFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType,
         mType,
         linvType,
         c10::ScalarType::Int,
         c10::ScalarType::Float,
         c10::ScalarType::Float});

    RUN_TUPLE_MAYBE_WITH_ACC_THREAD(fp8_sdpa_recomp_fwd, hpu_op)
  }
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_fwd_lazy(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor> d_scale_q,
    const c10::optional<at::Tensor> d_scale_k,
    const c10::optional<at::Tensor> d_scale_v,
    const c10::optional<at::Tensor> q_scale_s,
    const c10::optional<at::Tensor> q_scale_o,
    const c10::optional<at::Tensor> d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  auto fwdOutType = q.scalar_type();
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn &&
      (!q_scale_o.has_value()))
    fwdOutType = at::ScalarType::BFloat16;

  return fp8_sdpa_recomp_fwd_common<c10::optional<at::Tensor>>(
      q,
      k,
      v,
      attention_mask,
      p,
      scale,
      is_causal,
      requires_backward,
      softmax_mode,
      d_scale_q,
      d_scale_k,
      d_scale_v,
      q_scale_s,
      q_scale_o,
      d_scale_s,
      is_amax_s,
      is_amax_o,
      valid_seq_len,
      seq_padding_type,
      fwdOutType);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_fwd_scalar_lazy(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    const double d_scale_q,
    const double d_scale_k,
    const double d_scale_v,
    const double q_scale_s,
    const double q_scale_o,
    const double d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  auto fwdOutType = q.scalar_type();
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn && (q_scale_o == 0.))
    fwdOutType = at::ScalarType::BFloat16;

  return fp8_sdpa_recomp_fwd_common<double>(
      q,
      k,
      v,
      attention_mask,
      p,
      scale,
      is_causal,
      requires_backward,
      softmax_mode,
      d_scale_q,
      d_scale_k,
      d_scale_v,
      q_scale_s,
      q_scale_o,
      d_scale_s,
      is_amax_s,
      is_amax_o,
      valid_seq_len,
      seq_padding_type,
      fwdOutType);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_fwd_lazy(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& d_scale_q,
    const c10::optional<at::Tensor>& d_scale_k,
    const c10::optional<at::Tensor>& d_scale_v,
    const c10::optional<at::Tensor>& q_scale_s,
    const c10::optional<at::Tensor>& q_scale_o,
    const c10::optional<at::Tensor>& d_scale_s,
    const bool is_amax_s,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  auto fwdOutType = q.scalar_type();
  auto sfmxType = q.scalar_type();
  // if (is_amax_s) {
  // sfmxType = c10::ScalarType::BFloat16;
  // }

  // Normally SDPA FWD Fp8 dtype is e4m3. Supporting
  // e5m2 for experiments
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn ||
      q.scalar_type() == at::ScalarType::Float8_e5m2) {
    if (q_scale_o.has_value()) {
      fwdOutType = q.scalar_type();
    } else {
      fwdOutType = at::ScalarType::BFloat16;
    }
  }

  if (p > 0.0) {
    c10::optional<Generator> gen;
    auto seed = habana::get_seed_tensor_hpu(gen);
    LazyOp<std::tuple<Tensor, Tensor, Tensor, Tensor>> hpu_op{
        "hpu::fp8_sdpa_fwd_dropout_seed",
        {seed,
         q,
         k,
         v,
         attention_mask,
         p,
         scale,
         is_causal,
         softmax_mode,
         d_scale_q,
         d_scale_k,
         d_scale_v,
         q_scale_s,
         q_scale_o,
         d_scale_s,
         is_amax_s,
         valid_seq_len,
         seq_padding_type},
        Fp8SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType, sfmxType, c10::ScalarType::Char, c10::ScalarType::Float});

    RUN_TUPLE_MAYBE_WITH_ACC_THREAD(fp8_sdpa_fwd, hpu_op)
  } else {
    LazyOp<std::tuple<Tensor, Tensor, Tensor, Tensor>> hpu_op{
        "hpu::fp8_sdpa_fwd",
        {q,
         k,
         v,
         attention_mask,
         p,
         scale,
         is_causal,
         softmax_mode,
         d_scale_q,
         d_scale_k,
         d_scale_v,
         q_scale_s,
         q_scale_o,
         d_scale_s,
         is_amax_s,
         valid_seq_len,
         seq_padding_type},
        Fp8SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType, sfmxType, c10::ScalarType::Char, c10::ScalarType::Float});

    RUN_TUPLE_MAYBE_WITH_ACC_THREAD(fp8_sdpa_fwd, hpu_op)
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_bwd_lazy(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& P,
    const c10::optional<at::Tensor>& dm,
    const bool is_causal,
    const double p,
    const double scale,
    const at::Tensor& fwd_out) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<std::tuple<Tensor, Tensor, Tensor>> hpu_op{
      "hpu::sdpa_bwd",
      {grad, q, k, v, P, dm, is_causal, p, scale, fwd_out},
      SDPABwdOutputShape};

  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(sdpa_bwd, hpu_op)
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_bwd_lazy(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& P,
    const c10::optional<at::Tensor>& dm,
    const bool is_causal,
    const double p,
    const double scale,
    const c10::optional<at::Tensor>& d_scale_q,
    const c10::optional<at::Tensor>& d_scale_k,
    const c10::optional<at::Tensor>& d_scale_v,
    const c10::optional<at::Tensor>& d_scale_s,
    const c10::optional<at::Tensor>& d_scale_do,
    const c10::optional<at::Tensor>& d_scale_ds,
    const c10::optional<at::Tensor>& q_scale_s,
    const c10::optional<at::Tensor>& q_scale_ds,
    const bool is_amax_ds,
    const at::Tensor& fwd_out) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<std::tuple<Tensor, Tensor, Tensor, Tensor>> hpu_op{
      "hpu::fp8_sdpa_bwd",
      {grad,
       q,
       k,
       v,
       P,
       dm,
       is_causal,
       p,
       scale,
       d_scale_q,
       d_scale_k,
       d_scale_v,
       d_scale_s,
       d_scale_do,
       d_scale_ds,
       q_scale_s,
       q_scale_ds,
       is_amax_ds,
       fwd_out},
      Fp8SDPABwdOutputShape};

  // Set grad type to BF16 for now
  auto gradType = c10::ScalarType::BFloat16;
  hpu_op.set_scalar_types(
      {gradType, // dQ
       gradType, // dK
       gradType, // dV
       c10::ScalarType::Float}); // amax_ds

  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(fp8_sdpa_bwd, hpu_op)
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> sdpa_recomp_fwd_lazy(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  if (p > 0.0) {
    c10::optional<Generator> gen;
    auto seed = habana::get_seed_tensor_hpu(gen);
    LazyOp<std::tuple<Tensor, Tensor, Tensor, Tensor>> hpu_op{
        "hpu::sdpa_recomp_fwd_dropout_seed",
        {seed,
         q,
         k,
         v,
         attention_mask,
         p,
         scale,
         is_causal,
         requires_backward,
         softmax_mode,
         valid_seq_len,
         seq_padding_type},
        SDPARecompFwdOutputShape};
    hpu_op.set_scalar_types(
        {q.scalar_type(),
         q.scalar_type(),
         c10::ScalarType::Float,
         c10::ScalarType::Int});
    RUN_TUPLE_MAYBE_WITH_ACC_THREAD(sdpa_recomp_fwd, hpu_op)
  } else {
    LazyOp<std::tuple<Tensor, Tensor, Tensor, Tensor>> hpu_op{
        "hpu::sdpa_recomp_fwd",
        {q,
         k,
         v,
         attention_mask,
         p,
         scale,
         is_causal,
         requires_backward,
         softmax_mode,
         valid_seq_len,
         seq_padding_type},
        SDPARecompFwdOutputShape};
    auto linvType = c10::ScalarType::Float;

    if ((softmax_mode == "fast") &&
        (q.scalar_type() == c10::ScalarType::BFloat16)) {
      linvType = c10::ScalarType::BFloat16;
    }
    hpu_op.set_scalar_types(
        {q.scalar_type(), q.scalar_type(), linvType, c10::ScalarType::Int});
    RUN_TUPLE_MAYBE_WITH_ACC_THREAD(sdpa_recomp_fwd, hpu_op)
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_recomp_bwd_lazy(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const at::Tensor& m,
    const at::Tensor& linv,
    const c10::optional<at::Tensor>& seed,
    const bool is_causal,
    const double p,
    const double scale,
    c10::string_view softmax_mode,
    const at::Tensor& fwd_out) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<std::tuple<Tensor, Tensor, Tensor>> hpu_op{
      "hpu::sdpa_recomp_bwd",
      {grad,
       q,
       k,
       v,
       attention_mask,
       m,
       linv,
       seed,
       is_causal,
       p,
       scale,
       softmax_mode,
       fwd_out},
      SDPARecompBwdOutputShape};

  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(sdpa_recomp_bwd, hpu_op)
}
at::Tensor scaled_triangular_softmax_lazy(
    const at::Tensor& self,
    double inv_scale_attn,
    const c10::optional<at::Tensor>& exp_sum_recpr,
    const c10::optional<at::Tensor>& max) {
  LazyOp<at::Tensor> op{
      "hpu::scaled_triangular_softmax",
      {self, inv_scale_attn, exp_sum_recpr, max},
      {{self.sizes().vec()}}};

  RUN_MAYBE_WITH_ACC_THREAD(scaled_triangular_softmax, op)
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
scaled_triangular_softmax_retain_lazy(
    const at::Tensor& self,
    double inv_scale_attn) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  auto out_shape = self.sizes().vec();
  auto retain_output_shape = out_shape;
  retain_output_shape.back() = 1;
  LazyOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> op{
      "hpu::scaled_triangular_softmax_retain",
      {self, inv_scale_attn},
      {{out_shape, retain_output_shape, retain_output_shape}}};
  op.set_scalar_types(
      {self.scalar_type(), c10::ScalarType::Float, self.scalar_type()});

  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(scaled_triangular_softmax_retain, op)
}

at::Tensor& kv_reorder_lazy(
    at::Tensor& self,
    const at::Tensor start,
    const at::Tensor end,
    const at::Tensor beam_idx) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor&> op{
      "hpu::kv_reorder_", {self, start, end, beam_idx}, {{self.sizes().vec()}}};

  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(kv_reorder, op, self)
}

at::Tensor scaled_masked_triangular_softmax_lazy(
    const at::Tensor& self,
    const at::Tensor& start_end,
    double inv_scale_attn,
    int64_t grouped_batch_size,
    bool use_max,
    int64_t mode,
    c10::optional<at::ScalarType> out_dtype) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op{
      "hpu::scaled_masked_triangular_softmax",
      {self,
       start_end,
       inv_scale_attn,
       grouped_batch_size,
       use_max,
       mode,
       out_dtype},
      {{self.sizes().vec()}}};
  op.set_scalar_types({out_dtype.value_or(self.scalar_type())});

  RUN_MAYBE_WITH_ACC_THREAD(scaled_masked_triangular_softmax, op)
}

at::Tensor& in_place_interleave_lazy(at::Tensor& self) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;

  LazyOp<at::Tensor&> op{
      "hpu::in_place_interleave_", {self}, {{self.sizes().vec()}}};

  RUN_INPLACE_MAYBE_WITH_ACC_THREAD(in_place_interleave, op, self)
}

} // namespace habana_lazy
