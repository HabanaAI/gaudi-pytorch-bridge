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
#include <cstdlib>
#include <ctime>
#include <utility>
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/aten_hpu_type_default.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/bitwise_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/conv_kernels.h"
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/embedding_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_kernels/loss_kernels.h"
#include "habana_kernels/nonzero_kernel.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/pool_kernels.h"
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
#include "habana_lazy/ops/optimizer_sparse_sgd_with_valid_count.h"
#include "habana_lazy/ops/pool.h"
#include "habana_lazy/ops/reduce_ops.h"
#include "habana_lazy/ops/shape_ops.h"
#include "habana_lazy/ops/softmax.h"
#include "habana_lazy/ops/tensor_shape.h"
#include "habana_lazy/ops/topk.h"
#include "habana_lazy/ops/unpack.h"
#include "habana_lazy/ops/upsample.h"
#include "habana_lazy/view.h"
#include "hpu_ops/generated/hpu_op.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/synapse_helpers/util.h"

using namespace habana;
using namespace at;

namespace habana_lazy {
static std::vector<int64_t> device_shape_tensor_size = {SYN_MAX_TENSOR_DIM};

#define STRINGIFY(op_code) #op_code

#define HPU_LAZY_FUNC_NAME(op_code) op_code##_hpu_lazy
#define HPU_LAZY_FUNC_NAME_INPLACE(op_code) op_code##hpu_lazy_
#define HPU_LAZY_WRAP_KERNEL(op_code)                       \
  Tensor HPU_LAZY_FUNC_NAME(op_code)(const Tensor& self) {  \
    PT_LAZY_TRACE;                                          \
    LazyOp<at::Tensor> k{STRINGIFY(aten::op_code), {self}}; \
    return k.call();                                        \
  }
#define HPU_LAZY_WRAP_KERNEL_INPLACE(op_code)                  \
  Tensor& HPU_LAZY_FUNC_NAME_INPLACE(op_code)(Tensor & self) { \
    PT_LAZY_TRACE;                                             \
    LazyOp<at::Tensor&> k{STRINGIFY(aten::op_code), {self}};   \
    return k.call(self);                                       \
  }

bool to_lower_as_strided() {
  return GET_ENV_FLAG_NEW(PT_HPU_LOWER_AS_STRIDED);
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
void flush_op(at::TensorList tensors, size_t lazy_eager_key) {
  const bool m_flush_op = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2;
  const bool m_random_flush = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 3;
  DebugHelper::getInstance().incrementAccumulatedOps();

  if (m_flush_op) {
    std::vector<HbLazyTensor> hl_tensors;
    hl_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
      hl_tensors.push_back(GetHbLazyTensor(t));
    }
    HbLazyTensor::SyncTensorsGraph(&hl_tensors, lazy_eager_key);
  } else if (m_random_flush) {
    flushWithMarkStep();
  } else if (DebugHelper::getInstance().isExceededMaxAccumlatedSize()) {
    PT_LAZY_DEBUG("Reached max accumulated graph size, triggering a mark_step");
    HbLazyTensor::StepMarker({});
  }
}

at::Tensor preProcessIfLongorDouble(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool& processed) {
  at::Tensor processed_tensor_cpu;
  c10::ScalarType old_type = src.scalar_type();
  c10::ScalarType new_type = src.scalar_type();
  // We need to cast data on CPU before copying if there is some unsupported
  // type
  if (src.scalar_type() == c10::ScalarType::Long) {
    processed_tensor_cpu = src.to(c10::ScalarType::Int);
    processed = true;
    old_type = c10::ScalarType::Long;
    new_type = c10::ScalarType::Int;
  } else if (src.scalar_type() == c10::ScalarType::Double) {
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

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VIEW_TABLE) == false) {
    auto view = hl_dst.getView();
    // FIXME: deactivating code to add control edge for updating views of the
    // tensors
    // This case isnt hit right now and ww got cache crashes with this control
    // edge needs a design review and fix to activate
    if (view) {
      if (!to_lower_as_strided()) {
        PT_LAZY_DEBUG(
            "WARNING: We are hitting a case where the dst tensor has a view. Not all cases are covered so functionality might be impacted ");
        return;
      }
      // This is how we create the control edge -
      // There is a view on the dst, which means this is a view of
      // another tensor src, that was created the following way -
      //    dst = as_strided(src)
      //
      // We now have an op that is writing to dst. Since dst
      // is a view on src, we add a control edge here so that -
      //    src = control_edge_(dst, src)
      //
      // This will ensure all future ops on view_tensor will be scheduled
      // after this op updating dst
      AddControlEdge(dst, view->getAtTensor());
    }
  }
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

Tensor add_strided_insert_node(
    const Tensor& orig_t,
    const Tensor& insert_t,
    IntArrayRef strides,
    int64_t offset,
    bool is_flush = true) {
  auto mf = orig_t.suggest_memory_format();
  ir::NodePtr node;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    std::string node_str = ((mf == c10::MemoryFormat::ChannelsLast) ||
                            (mf == c10::MemoryFormat::ChannelsLast3d))
        ? "hpu::strided_insert_cl_ds"
        : "hpu::strided_insert_ds";

    auto out_stride_st = empty_hpu_lazy(
        strides,
        orig_t.options(),
        c10::MemoryFormat::Contiguous,
        false,
        SHAPE_TENSOR);

    std::vector<int64_t> offset_vec = {offset};
    IntArrayRef offset_ref(offset_vec.data(), offset_vec.size());
    auto offset_st = empty_hpu_lazy(
        offset_ref,
        orig_t.options(),
        c10::MemoryFormat::Contiguous,
        false,
        SHAPE_TENSOR);

    node = std::make_shared<ir::StridedInsert>(
        orig_t, insert_t, out_stride_st, offset_st, node_str);
  } else {
    std::string node_str = ((mf == c10::MemoryFormat::ChannelsLast) ||
                            (mf == c10::MemoryFormat::ChannelsLast3d))
        ? "hpu::strided_insert_cl"
        : "hpu::strided_insert";

    node = std::make_shared<ir::StridedInsert>(
        orig_t, insert_t, strides, offset, node_str);
  }
  auto result = empty_hpu_lazy(
      orig_t.sizes(), orig_t.options(), orig_t.suggest_memory_format(), false);
  auto hl_result = GetHbLazyTensor(result);

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());

  if (is_flush) {
    flush_op(result);
  }
  return result;
}

void strided_insert_hpu_lazy(
    const Tensor& self,
    const Tensor& insert_t,
    bool is_flush) {
  PT_LAZY_TRACE;
  auto hl_self = GetHbLazyTensor(self);
  auto id = hl_self.getTensorUniqueId();

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  TORCH_CHECK(
      context->view_table.find(id) != context->view_table.end(),
      "incorrect tensor id");
  StrideParams* params_ptr = &context->view_table[id];

  // pick the most recent version
  Tensor recent_orig_t = get_recent_base_tensor(params_ptr->t);
  auto out = add_strided_insert_node(
      recent_orig_t,
      insert_t,
      params_ptr->strides,
      params_ptr->offset,
      is_flush);

  // update orig tensor map
  auto param_id = GetHbLazyTensor(params_ptr->t).getTensorUniqueId();
  context->orig_tensor_map[param_id] = out;
  return;
}

Tensor get_parent_tensor(const Tensor& self) {
  // Handle multi level views by traversing up to reach the base tensor (i.e.
  // until there is no entry in view table)
  auto out = self;
  auto id = GetHbLazyTensor(self).getTensorUniqueId();

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  // handle multi level views
  while (context->view_table.find(id) != context->view_table.end()) {
    out = context->view_table[id].t;
    id = GetHbLazyTensor(out).getTensorUniqueId();
  }

  return out;
}

const Tensor& get_recent_base_tensor(const Tensor& self) {
  /* Fetch the most recent version of base from orig tensor map*/
  auto id = GetHbLazyTensor(self).getTensorUniqueId();

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);

  if (context->orig_tensor_map.find(id) != context->orig_tensor_map.end()) {
    return context->orig_tensor_map[id];
  }

  return self;
}

bool HandleViews(const Tensor& t, const HbLazyTensor& hl_t) {
  PT_LAZY_TRACE;
  bool is_view = false;
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto id = hl_t.getTensorUniqueId();
  auto it = context->view_table.find(id);
  if (it != context->view_table.end()) {
    StrideParams& params = context->view_table[id];

    // pick the most recent version
    auto recent_orig_t = get_recent_base_tensor(params.t);

    auto t_opt = c10::make_optional(t);

    Tensor out = add_strided_view_node(
        recent_orig_t,
        params.sizes,
        params.strides,
        params.offset,
        false,
        t_opt);

    is_view = true;
  }
  return is_view;
}

habana_lazy::HbLazyTensor HandleViewsOrUpdate(
    const at::Tensor& t,
    habana_lazy::HbLazyTensor& hl_t) {
  auto hl_out = hl_t;
  auto is_view = HandleViews(t, hl_t);

  if (is_view == false) {
    auto t_updated = get_recent_base_tensor(t);
    hl_out = GetHbLazyTensor(t_updated);
  }
  return hl_out;
}

std::vector<Tensor> HandleViewsTensorList(const TensorList& in_list) {
  std::vector<Tensor> updated_t_list;

  for (auto t : in_list) {
    auto hl_t = GetHbLazyTensor(t);

    auto is_view = HandleViews(t, hl_t);

    if (is_view == false) {
      auto t_updated = get_recent_base_tensor(t);
      updated_t_list.push_back(t_updated);
    } else {
      updated_t_list.push_back(t);
    }
  }

  return updated_t_list;
}

Tensor HandleViewsD2H(const Tensor& src) {
  PT_LAZY_TRACE;
  auto out = src;
  auto hl_t = GetHbLazyTensor(src);

  auto is_view = HandleViews(src, hl_t);

  if (is_view) {
    hl_t = GetHbLazyTensor(src);
    std::vector<HbLazyTensor> tensors = {hl_t};
    HbLazyTensor::SyncTensorsGraph(&tensors);
  } else {
    // check for updated version
    out = get_recent_base_tensor(src);
  }

  return out;
}

bool HandleViewsD2D(const at::Tensor& src, const at::Tensor& dst) {
  PT_LAZY_TRACE;
  bool is_view = false;

  auto hb_src = GetHbLazyTensor(src);
  HandleViews(src, hb_src);

  // support for lhs sliced insert ex: a[::] = b. PT lowers this op as
  // as_strided + d2d copy. we replace both these ops by strided insert. This
  // way strided tensor support for D2D copy is avoided
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto hlresult = GetHbLazyTensor(dst);
  auto id = hlresult.getTensorUniqueId();
  auto it = context->view_table.find(id);

  if (it != context->view_table.end()) {
    is_view = true;

    StrideParams* params_ptr = &context->view_table[id];

    // get the base tensor
    // check for most recent version of the original tensor
    auto orig_t = get_parent_tensor(params_ptr->t);
    auto orig_t_id = GetHbLazyTensor(orig_t).getTensorUniqueId();

    auto src_parent = get_parent_tensor(src);
    auto src_parent_id = GetHbLazyTensor(src_parent).getTensorUniqueId();

    // the id check avoids a cycle with strided insert node
    // scenario t1_h[i - 1] += 1. Here the output of the add can be used
    // directly instead of performing one more strided insert
    if (src_parent_id != orig_t_id) {
      auto recent_orig_t = get_recent_base_tensor(orig_t);
      auto out = add_strided_insert_node(
          recent_orig_t, src, params_ptr->strides, params_ptr->offset);

      // update orig tensor map
      context->orig_tensor_map[orig_t_id] = out;
    }
  }

  return is_view;
}

void updateViewTable(HbLazyTensor& hl_view_t, StrideParams& params) {
  PT_LAZY_TRACE;
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto id = hl_view_t.getTensorUniqueId();
  context->view_table[id] = params;
}

/**
 * Returns a tensor for a scalar value.
 * In case of 64b dtypes such as Long/Double it returns a tensor
 * where the FE (user's PT tensor) is in Long/Double and the BE (device storage)
 * is in Int/Float
 */
at::Tensor get_tensor_for_scalar(
    float alpha,
    const at::TensorOptions& options) {
  at::Tensor alpha_tensor;

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);

  auto map_it = context->scalar_to_tensor_map.find(
      std::make_pair(alpha, options.dtype().toScalarType()));
  if (map_it == context->scalar_to_tensor_map.end()) {
    alpha_tensor = at::tensor(alpha).to(options.dtype()).to(c10::kHPU, true);
    context->scalar_to_tensor_map[std::make_pair(
        alpha, options.dtype().toScalarType())] = alpha_tensor;
  } else {
    alpha_tensor = map_it->second;
  }

  return alpha_tensor;
}

Tensor& copy_hpu_lazy_D2D(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;
  auto is_5d_tensor = self.dim() == 5;
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
  Tensor src_updated = get_recent_base_tensor(src);
  HbLazyTensor hb_tensor =
      GetOrCreateHbLazyTensor(src_updated, src_updated.device());
  auto hlresult = GetOrCreateHbLazyTensor(self, src_updated.device());
  auto layout_format = hb_tensor.GetTensorLayout();
  hlresult.SetTensorLayout(layout_format);
  bool permuted = false;
  /* We can't create a long/double target in the device. Even a cast will not
    work as these data types are not available within the device. The only way
    to make progress is to just do a normal D2D so that the target will also
    be the same as source, and when we want to pull this out to CPU, the D2H
    will handle the type conversion*/
  if ((self.scalar_type() == c10::ScalarType::Long) ||
      (self.scalar_type() == c10::ScalarType::Double) ||
      (src_updated.dtype() == self.dtype())) {
    // If both src and dst are already processed ,  go and do the DMA dont
    // wait Else , If we already have storage in dst, add memcopy node to lazy
    // graph and we want to copy to existing tensor and not a new one
    // Kernel expects us to pass dst as second input in that case
    auto result_data = hlresult.CurrentTensorData();
    auto src_data = hb_tensor.CurrentTensorData();
    if (copy_transpose_valid(self, src)) {
      permuted = true;
      if (is_5d_tensor) {
        int64_t dim_chl_pos[] = {0, 2, 3, 4, 1};
        at::IntArrayRef chl_pos = dim_chl_pos;
        self = permute_cl_hpu_lazy(src, chl_pos);
      } else {
        int64_t dim_chl_pos[] = {0, 2, 3, 1};
        at::IntArrayRef chl_pos = dim_chl_pos;
        self = permute_cl_hpu_lazy(src, chl_pos);
      }
    } else if (!permuted) {
      auto src_id = hb_tensor.getTensorUniqueId();
      auto dst_id = hlresult.getTensorUniqueId();
      if (src_id == dst_id) {
        return self;
      }

      // graph cycle happens in squad 8x with view table mechanism
      // %id:3646 = hpu::as_strided_lazy(%id:18.1, %89, %90, %91)
      // %id:18 = hpu::habana_d2d_memcpy_other(%id:3646, %id:18.1)
      auto src_parent = get_parent_tensor(src_updated);
      auto src_parent_id = GetHbLazyTensor(src_parent).getTensorUniqueId();

      if (src_parent_id == dst_id) {
        return self;
      }

      // Handle views and lhs slice
      auto is_view = HandleViewsD2D(src, self);
      if (is_view == false) {
        if (!hlresult.CurrentIrValue().IsHpuInputNode()) {
          // add control edge to avoid GC error " writing to already
          // registered graph output"
          updateDstDependencies(hlresult, self, true);
        }

        AddMemcpy(src_updated, self);
        updateDstDependencies(hlresult, self);
      }
    }
  } else {
    node = std::make_shared<ir::Cast>(src, self.scalar_type(), non_blocking);

    auto context = habana_lazy_executor.getDeviceExecutionContext(0);
    auto id = GetHbLazyTensor(self).getTensorUniqueId();

    if (context->view_table.find(id) != context->view_table.end()) {
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

      HandleViewsD2D(src_cast, self);
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

Tensor maybe_contiguous(const Tensor& self) {
  auto output = self;
  if (!self.is_contiguous()) {
    if ((self.suggest_memory_format() != c10::MemoryFormat::ChannelsLast) &&
        (self.suggest_memory_format() != c10::MemoryFormat::ChannelsLast3d)) {
      PT_LAZY_DEBUG("Changing src tensor to contiguous before D2H");
      output = self.contiguous();
    }
  }

  return output;
}

Tensor as_strided_layout_hpu_lazy(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride) {
  int64_t dim_out_pos[] = {2, 3, 1, 0};
  int64_t dim_out_pos_3d[] = {2, 3, 4, 1, 0};
  IntArrayRef dims_ = dim_out_pos;
  if (self.dim() == 5)
    dims_ = dim_out_pos_3d;
  auto node =
      std::make_shared<ir::Permute>(self, dims_, "hpu::as_strided_layout_");
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

Tensor& copy_hpu_lazy_D2H(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;

  // This situation should not occur
  // Throwing an exception here for now to catch any cases that arise
  TORCH_CHECK(
      IsHbLazyTensor(src),
      "Habana Lazy : trying to copy back a tensor which does not have a lazy tensor");

  auto is_5d_tensor = src.dim() == 5;

  // handle views
  auto src_view = HandleViewsD2H(src);

  auto src_hb_tensor = GetHbLazyTensor(src_view);
  auto src_hb_tensor_data = src_hb_tensor.GetHbLazyTensorData();
  if (!src_hb_tensor_data) {
    TORCH_CHECK(
        false,
        "Habana copy_hpu_lazy_: no storage tensor attached for copy lazy source");
  }
  if (!src_hb_tensor_data.value().has_storage()) {
    TORCH_CHECK(
        false,
        "Habana copy_hpu_lazy_: trying to copy from a storage less lazy tensor");
  }

  // Handle non-contiguous src
  // TODO is this needed anymore
  auto _src = maybe_contiguous(src_view);

  // If _src is a lazy tensor make sure the execution till the point of _src
  // If src is a lazy tensor make sure the execution till the point of src
  // getting flled has finished before we start copying
  auto hb_tensor = GetHbLazyTensor(_src);
  auto tensor_data = hb_tensor.GetHbLazyTensorData();
  auto hl_tensor_data = habana_lazy::GetHbInternalTensorImpl(*tensor_data);
  // weights HWCK -> NCHW
  if ((hl_tensor_data->GetTensorLayout() == habana_lazy::LayoutFormat::kHWCK) &&
      (habana_lazy::exec::OptPassCfg::GetInstance()
           ->IsEnabledWeightPermutePass())) {
    hl_tensor_data->SetTensorLayout(habana_lazy::LayoutFormat::kNCHW);
    auto sizes = src.sizes().vec();
    std::vector<int> out_pos = {2, 3, 1, 0};
    std::vector<long int> swapped_sizes = {
        sizes[out_pos[0]],
        sizes[out_pos[1]],
        sizes[out_pos[2]],
        sizes[out_pos[3]]};
    std::vector<int> out_pos_5d = {2, 3, 4, 1, 0};
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
      auto permute_tensor = permute_hpu_lazy(strided_tensor, {4, 3, 0, 1, 2});
      HbLazyTensor hb_tensor = GetHbLazyTensor(permute_tensor);
      tensor_data = hb_tensor.GetHbLazyTensorData();
    } else {
      auto new_strides =
          CalculateStrides(swapped_sizes, c10::MemoryFormat::Contiguous);
      auto strided_tensor =
          // as_strided_hpu_lazy(src, swapped_sizes, new_strides, 0);
          as_strided_layout_hpu_lazy(src, swapped_sizes, new_strides);
      auto permute_tensor = permute_hpu_lazy(strided_tensor, {3, 2, 0, 1});
      HbLazyTensor hb_tensor = GetHbLazyTensor(permute_tensor);
      tensor_data = hb_tensor.GetHbLazyTensorData();
    }
  }
  TORCH_CHECK(
      tensor_data, "Trying to copy from lazy tensor with no backend memory");
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
    self = copy_hpu_(self, tensor_data.value(), non_blocking);
    self = self.to(type);
  } else {
    self = copy_hpu_(self, tensor_data.value(), non_blocking);
  }
  // No need to CreateHbLazyTensor for self as it is on CPU
  return self;
}

Tensor& copy_hpu_lazy_H2D(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;
  bool processed = false;

  auto context =
      habana_lazy_executor.getDeviceExecutionContext(self.device().index());
  auto exec_mode = context->getExecutionMode();
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
      Tensor at_internal_tensor =
          AtenInternalHbTensor(std::move(storage_impl), self.dtype());
      // Setup the tensor sizes & strides for tensor with dim = 4, else for
      // now assuming contiguous
      if (4 == self.dim()) {
        at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
            src.sizes(),
            CalculateStrides(src.sizes(), src.suggest_memory_format()));
      } else if (5 == self.dim()) {
        at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
            src.sizes(),
            CalculateStrides5d(src.sizes(), src.suggest_memory_format()));
      } else {
        at_internal_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(
            src.sizes());
      }
      self_hb_tensor.SetTensorData(at_internal_tensor);
    }
  }
  auto new_tensor = preProcessIfLongorDouble(src, self, processed);

  // Get the internal tensor for copy kernel
  // First get the lazy tensor
  auto self_hb_tensor = GetOrCreateHbLazyTensor(self, self.device());

  // Set the tensor as input and mark as input
  setTensorAsInputNode(self_hb_tensor);
  context->MarkTensorStatus(
      self_hb_tensor.getTensorUniqueId(), LazyTensorExecutionStatus::kINPUT);

  // We need to mark this tensor as executed
  // As this will be an input coming from host side, its doesnt need further
  // execution and is ready for consumption as input
  auto self_hb_tensor_data = self_hb_tensor.GetHbLazyTensorData();
  // This is the internal tensor, it isn't a lazy tensor
  auto self_internal_tesor = self_hb_tensor_data.value();
  HABANA_ASSERT(!TryGetHbLazyTensor(self_internal_tesor));

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
  Tensor at_internal_tensor =
      AtenInternalHbTensor(c10::Storage(storage_impl->storage()), self.dtype());
  at_internal_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
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
    c10::optional<int64_t> storage_offset) {
  ir::NodePtr node = nullptr;

  std::vector<int64_t> out_size_vec;
  std::vector<int64_t> out_stride_vec;
  std::tie(out_size_vec, out_stride_vec) =
      AsStridedOperator::compute_output_shape(self, size, stride);
  IntArrayRef out_size(out_size_vec.data(), out_size_vec.size());
  IntArrayRef out_stride(out_stride_vec.data(), out_stride_vec.size());
  auto offset = storage_offset.value_or(self.storage_offset());
  auto mf = self.suggest_memory_format();
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VIEW_TABLE)) {
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
      std::string node_str = ((mf == c10::MemoryFormat::ChannelsLast) ||
                              (mf == c10::MemoryFormat::ChannelsLast3d))
          ? "hpu::strided_view_cl_ds"
          : "hpu::strided_view_ds";

      auto out_size_st = empty_hpu_lazy(
          out_size,
          self.options(),
          c10::MemoryFormat::Contiguous,
          false,
          SHAPE_TENSOR);
      auto out_stride_st = empty_hpu_lazy(
          out_stride,
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

      node = std::make_shared<ir::StridedView>(
          self, out_size_st, out_stride_st, offset_st, node_str);
      return node;
    } else {
      std::string node_str = ((mf == c10::MemoryFormat::ChannelsLast) ||
                              (mf == c10::MemoryFormat::ChannelsLast3d))
          ? "hpu::strided_view_cl"
          : "hpu::strided_view";
      node = std::make_shared<ir::StridedView>(
          self, out_size, out_stride, offset, node_str);
    }
  } else {
    if ((stride.size() == 0) ||
        ((out_stride_vec.size() > 0) &&
         (out_stride_vec[out_stride_vec.size() - 1] == 1))) {
      int64_t offset = storage_offset ? storage_offset.value() : 0;

      auto mf = self.suggest_memory_format();
      std::string node_str = ((mf == c10::MemoryFormat::ChannelsLast) ||
                              (mf == c10::MemoryFormat::ChannelsLast3d))
          ? "hpu::as_strided_lazy_cl_"
          : "hpu::as_strided_lazy_";

      node = std::make_shared<ir::AsStrided>(
          self, out_size, out_stride, offset, node_str);
    }
  }
  return node;
}

Tensor add_strided_view_node(
    const Tensor& self,
    IntArrayRef size_in,
    IntArrayRef stride_in,
    int64_t storage_offset,
    bool is_update_view,
    c10::optional<Tensor> out_t) {
  PT_LAZY_TRACE;
  IntArrayRef size = size_in;
  bool is_0d_tensor = false;
  std::vector<int64_t> initvec{1};
  if (size_in.size() == 0) {
    size = initvec;
    is_0d_tensor = true;
  }
  IntArrayRef stride = stride_in;
  if (stride_in.size() == 0) {
    stride = initvec;
  }

  auto context =
      habana_lazy_executor.getDeviceExecutionContext(self.device().index());

  // when we get a call from lowering, we create a storage based backend
  // tensor
  if (context != nullptr) {
    auto exec_mode = context->getExecutionMode();
    if (exec_mode == kLOWERING) {
      auto result = empty_as_strided_lazy(self, size, stride, storage_offset);
      if (is_0d_tensor) {
        result.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
      }
      return result;
    }
  }

  auto self_ = get_parent_tensor(self);
  // self_ = get_recent_base_tensor(self_);

  // auto hb_tensor = GetOrCreateHbLazyTensor(self);
  // auto src_data = hb_tensor.CurrentTensorData();

  // We only support contiguous chunks of data to be taken as strided
  // As Device doesnt support strided tensors we dont support that case
  Tensor result;
  if (out_t.has_value()) {
    // actual op building phase
    // use the actual out tensor provided by the inplace op
    result = out_t.value();
  } else {
    result = empty_strided_hpu_lazy(
        size, stride, self_.options(), false, DATA_TENSOR, storage_offset);
  }
  auto hb_result = GetHbLazyTensor(result);

  StrideParams params = {self_, size.vec(), stride.vec(), storage_offset};

  if (is_update_view) {
    updateViewTable(hb_result, params);
  } else {
    ir::Value& out = hb_result.CurrentIrValue();
    ir::NodePtr node =
        create_as_strided_node(params.t, size, stride, storage_offset);
    out.SetNode(
        node,
        hb_result.GetDevice(),
        hb_result.GetSizes(),
        hb_result.dtype_optional());
  }

  if (is_0d_tensor) {
    result.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }

  // when is_update_view == True is flush op is done in as_strided_hpu_lazy
  if (!is_update_view) {
    flush_op(result);
  }

  return result;
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

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VIEW_TABLE)) {
    // lazy within lazy. as strided node is not here. Only the view table update
    // happens here
    auto storage_offset_val = storage_offset.value_or(self.storage_offset());

    auto out = add_strided_view_node(
        self,
        size_in,
        stride_in,
        storage_offset_val,
        true /*is_update_view*/,
        c10::nullopt);
    auto context =
        habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
    if (context->getExecutionMode() != kLOWERING) {
      flush_op(out);
    }
    return out;
  } else {
    IntArrayRef size = size_in;
    bool is_0d_tensor = false;
    std::vector<int64_t> initvec{1};
    if (size_in.size() == 0) {
      size = initvec;
      is_0d_tensor = true;
    }
    IntArrayRef stride = stride_in;
    if (stride_in.size() == 0) {
      stride = initvec;
    }

    auto context =
        habana_lazy_executor.getDeviceExecutionContext(self.device().index());

    // when we get a call from lowering, we create a storage based backend
    // tensor
    if (context != nullptr) {
      auto exec_mode = context->getExecutionMode();
      if (exec_mode == kLOWERING) {
        auto result = empty_as_strided_lazy(self, size, stride, storage_offset);
        if (is_0d_tensor) {
          result.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
        }
        return result;
      }
    }

    auto hb_tensor = GetOrCreateHbLazyTensor(self);
    auto src_data = hb_tensor.CurrentTensorData();

    ir::NodePtr node =
        create_as_strided_node(self, size, stride, storage_offset);

    // We only support contiguous chunks of data to be taken as strided
    // As Device doesnt support strided tensors we dont support that case

    if (node != nullptr) {
      auto result = empty_strided_hpu_lazy(size, stride, self.options(), false);

      auto hb_result = GetHbLazyTensor(result);
      ir::Value& out = hb_result.CurrentIrValue();
      out.SetNode(
          node,
          hb_result.GetDevice(),
          hb_result.GetSizes(),
          hb_result.dtype_optional());
      // update the view if any
      updateDstDependencies(hb_result, result);
      // std::vector<at::Tensor> input_pt_vec{self};
      // node->AddInputPtTensors(input_pt_vec);
      // Add a view of the parent to the result so that its remembered
      // If this tensor is used as a dst in any op, we need to update the parent
      ir::LazyView view(self, hb_tensor.GetIrValue());
      hb_result.addView(view);
      // flush_op(result);
      if (is_0d_tensor) {
        result.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
      }
      return result;
    } else {
      TORCH_CHECK(0, "as_strided with FCD stride != 1 not supported");
    }
    flush_op(self);
    return self;
  }
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
    // update the view if any
    updateDstDependencies(hb_result, self);
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

  auto copy_node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::habana_d2d_memcpy_other"),
      {hl_src.GetIrValue(), hl_dst.GetIrValue()});

  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      dst.device().index());
  context->MarkTensorRegistered(hl_dst.getTensorUniqueId());
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

Tensor asin_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::asin", {self}};
  return k.call();
}

HPU_LAZY_WRAP_KERNEL(acos)
HPU_LAZY_WRAP_KERNEL_INPLACE(acos_)

Tensor acosh_hpu_lazy(const at::Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::acosh", {self}};
  return k.call();
}

Tensor& acosh_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::acosh_", {self}};
  return k.call(self);
}

Tensor asinh_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::asinh", {self}};
  return k.call();
}
Tensor& asinh_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::asinh_", {self}};
  return k.call(self);
}
Tensor atan_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::atan", {self}};
  return k.call();
}
Tensor& atan_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::atan_", {self}};
  return k.call(self);
}
Tensor atanh_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::atanh", {self}};
  return k.call();
}
Tensor& atanh_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::atanh_", {self}};
  return k.call(self);
}
Tensor cosh_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::cosh", {self}};
  return k.call();
}
Tensor& cosh_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::cosh_", {self}};
  return k.call(self);
}
Tensor sin_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::sin", {self}};
  return k.call();
}
Tensor cos_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::cos", {self}};
  return k.call();
}
Tensor& tanh_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::tanh_", {self}};
  return k.call(self);
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

Tensor view_hpu_lazy(const Tensor& self, IntArrayRef size) {
  PT_LAZY_TRACE;

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VIEW_TABLE)) {
    auto inferred_size = habana_helpers::infer_size(size, self.numel());
    auto stride =
        at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
    TORCH_CHECK(
        stride.has_value(),
        "view size is "
        "not compatible with input tensor's size and stride (at least one dimension"
        " spans across two contiguous subspaces). Use .reshape(...) instead.");
    auto stride_value = *stride;

    return as_strided_hpu_lazy(
        self, inferred_size, stride_value, self.storage_offset());
  } else {
    int64_t sum_elm = 1;
    for (auto& i : self.sizes()) {
      sum_elm *= i;
    }
    auto inferred_size =
        habana_helpers::infer_size(size, static_cast<int64_t>(sum_elm));

    ir::NodePtr node = std::make_shared<ir::View>(self, inferred_size);
    LazyOp<at::Tensor, ir::View> k{node, {self, size}, {inferred_size}};
    return k.call();
  }
}

Tensor addcmul_hpu_lazy(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::addcmul", {self, tensor1, tensor2, alpha}};
  return k.call();
}
Tensor& addcmul_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  if (!tensor1.is_same(tensor2)) {
    auto mul_out = mul_tensor_hpu_lazy(tensor1, tensor2);
    add_tensor_hpu_lazy_(self, mul_out, alpha);
  } else {
    // implement addcmul_ as add_(pow(tensor1,2), alpha)
    auto temp = pow_tensor_scalar_hpu_lazy(tensor1, 2.0);
    add_tensor_hpu_lazy_(self, temp, alpha);
  }

  flush_op(self);
  return self;
}
Tensor addcdiv_hpu_lazy(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::addcdiv", {self, tensor1, tensor2, alpha}};
  return k.call();
}

Tensor& addcdiv_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  auto alpha_float = alpha.toFloat();
  if (alpha_float == 1.0) {
    LazyOp<at::Tensor&> op{"aten::addcdiv_", {self, tensor1, tensor2, alpha}};
    return op.call(self);
  } else {
    auto div_out = div_tensor_hpu_lazy(tensor1, tensor2);
    auto alpha_tensor = get_tensor_for_scalar(alpha_float, div_out.options());
    auto mul_out = mul_tensor_hpu_lazy(alpha_tensor, div_out);
    auto out = add_tensor_hpu_lazy_(self, mul_out, 1.0);
  }

  flush_op(self);
  return self;
}

Tensor add_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  auto alpha_float = alpha.toFloat();

  if (alpha_float != 1.0) {
    at::Tensor alpha_tensor =
        get_tensor_for_scalar(alpha_float, other.options());

    auto hl_alpha = GetOrCreateHbLazyTensor(alpha_tensor, c10::kHPU);
    auto mul_out = mul_tensor_hpu_lazy(other, alpha_tensor);
    return add_tensor_hpu_lazy(self, mul_out, 1.0);
  } else {
    LazyBinaryOp<at::Tensor> k{
        "aten::add",
        {self, other, alpha},
        {},
        {BinaryOperator::compute_output_shape(self, other)}};
    k.ConvertWrappedTensorToScalar();
    return k.call();
  }
}

Tensor add_scalar_hpu_lazy(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> op{
      "aten::add", {self, other, alpha}, {}, {self.sizes().vec()}};
  return op.call();
}

Tensor& add_scalar_hpu_lazy_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  auto other_tensor = get_tensor_for_scalar(other.toFloat(), self.options());

  return add_tensor_hpu_lazy_(self, other_tensor, alpha);
}

Tensor& add_tensor_hpu_lazy_(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  auto alpha_float = alpha.toFloat();
  if (alpha_float != 1.0) {
    at::Tensor alpha_tensor =
        get_tensor_for_scalar(alpha_float, other.options());

    auto hl_alpha = GetOrCreateHbLazyTensor(alpha_tensor, c10::kHPU);
    auto mul_out = mul_tensor_hpu_lazy(other, alpha_tensor);
    return add_tensor_hpu_lazy_(self, mul_out, 1.0);
  } else {
    LazyBinaryOp<Tensor&> op("aten::add_", {self, other, alpha});
    op.ConvertWrappedTensorToScalar();
    return op.call(self);
  }
}

Tensor sub_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  LazyBinaryOp<at::Tensor> k{
      "aten::sub",
      {self, other, alpha},
      {},
      {BinaryOperator::compute_output_shape(self, other)}};
  return k.call();
}

Tensor& sub_tensor_hpu_lazy_(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;

  LazyBinaryOp<Tensor&> op("aten::sub_", {self, other, alpha});
  return op.call(self);
}

Tensor sub_scalar_hpu_lazy(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> op{
      "aten::sub", {self, other, alpha}, {}, {self.sizes().vec()}};
  return op.call();
}

Tensor& sub_scalar_hpu_lazy_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  auto other_tensor = get_tensor_for_scalar(other.toFloat(), self.options());
  return sub_tensor_hpu_lazy_(self, other_tensor, alpha);
}
Tensor rsub_scalar_hpu_lazy(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> op("aten::rsub", {self, other, alpha});
  return op.call();
}
Tensor& mul_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;

  LazyBinaryOp<Tensor&> op("aten::mul_", {self, other});
  return op.call(self);
}

Tensor where_tensor_hpu_lazy(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k(
      "aten::_s_where",
      {condition, self, other},
      {},
      {},
      1 /*output metadata is picked from self*/);
  return k.call();
}

Tensor mul_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyBinaryOp<at::Tensor> k{
      "aten::mul",
      {self, other},
      {},
      {BinaryOperator::compute_output_shape(self, other)}};
  return k.call();
}

Tensor& mul_out_hpu_lazy(Tensor& out, const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  // 8x all reduce optimization to avoid out variant that requires tensor with
  // storage. //TODO enhance lazy op framework to convert out variant to out of
  // place variant
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
  auto id = GetHbLazyTensor(out).getTensorUniqueId();
  if (context->view_table.find(id) != context->view_table.end()) {
    auto orig_out = out;
    auto temp = mul_tensor_hpu_lazy(self, other);
    strided_insert_hpu_lazy(orig_out, temp);
  } else {
    std::vector<at::Tensor> metatens_tensors = {self, other, out};
    auto metatens = habana::GetMetaTensorList(metatens_tensors);
    at::TensorList metavar =
        at::mul_outf(metatens[0], metatens[1], metatens[2]);
    LazyOp<at::Tensor&> hpu_op{"aten::mul", {self, other, out}, metavar};
    return hpu_op.call(out);
  }

  return out;
}

Tensor mul_scalar_hpu_lazy(const Tensor& self, const Scalar& other) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k("aten::mul", {self, other});
  return k.call();
}

Tensor& mul_scalar_hpu_lazy_(Tensor& self, const Scalar& other) {
  PT_LAZY_TRACE;

  auto other_tensor = get_tensor_for_scalar(other.toFloat(), self.options());
  LazyOp<at::Tensor&> k("aten::mul_", {self, other_tensor});
  return k.call(self);
}

Tensor div_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;

  LazyBinaryOp<at::Tensor> k{
      "aten::div",
      {self, other},
      {},
      {BinaryOperator::compute_output_shape(self, other)}};
  return k.call();
}
Tensor& div_tensor_hpu_lazy_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{
      "hpu::div_out",
      {out, self, other},
      {},
      {BinaryOperator::compute_output_shape(self, other)}};
  return k.call(out);
}

Tensor& div_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;

  LazyBinaryOp<at::Tensor&> k{"aten::div_", {self, other}};
  return k.call(self);
}

Tensor div_scalar_hpu_lazy(const Tensor& self, const Scalar& other) {
  PT_LAZY_TRACE;

  auto other_tensor = get_tensor_for_scalar(other.toFloat(), self.options());
  LazyOp<at::Tensor> k{"aten::div", {self, other_tensor}};
  return k.call();
}

Tensor& div_scalar_hpu_lazy_(Tensor& self, const Scalar& other) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor&> k{"aten::div_", {self, other}};
  return k.call(self);
}

Tensor pow_tensor_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyBinaryOp<at::Tensor> k{
      "aten::pow",
      {self, other},
      {},
      {BinaryOperator::compute_output_shape(self, other)}};
  return k.call();
}

Tensor& pow_tensor_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyBinaryOp<Tensor&> k("aten::pow_", {self, other});
  return k.call(self);
}

Tensor pow_tensor_scalar_hpu_lazy(const Tensor& self, const Scalar& other) {
  PT_LAZY_TRACE;
  if (other.equal(2)) {
    return mul_tensor_hpu_lazy(self, self);
  } else if (other.equal(3)) {
    auto temp = mul_tensor_hpu_lazy(self, self);
    return mul_tensor_hpu_lazy(temp, self);
  } else {
    LazyOp<at::Tensor> k{"aten::pow", {self, other}};
    return k.call();
  }
}

Tensor& pow_tensor_scalar_hpu_lazy_(Tensor& self, const Scalar& other) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::pow_", {self, other}};
  return k.call(self);
}

Tensor pow_scalar_tensor_hpu_lazy(const Scalar& other, const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::pow", {other, self}, {}, {self.sizes().vec()}, 1};
  return k.call();
}

Tensor all_dim_hpu_lazy(const Tensor& self, int64_t dim, bool keepdim) {
  PT_LAZY_TRACE;
  ir::NodePtr node = std::make_shared<ir::AllDim>(self, dim, keepdim);
  std::vector<int64_t> sizes =
      ReduceOperator::compute_output_shape(self, dim, keepdim);
  LazyOp<at::Tensor, ir::AllDim> op(node, {self, dim, keepdim}, {sizes});
  return op.call();
}

Tensor permute_wt_hpu(const Tensor& self) {
  Tensor result = self;
  if (habana_lazy::exec::OptPassCfg::GetInstance()
          ->IsEnabledWeightPermutePass() &&
      (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1)) {
    if (self.dim() == 4 || self.dim() == 5) {
      auto hb_tensor = GetOrCreateHbLazyTensor(self, self.device());
      auto layout_format = hb_tensor.GetTensorLayout();
      ir::NodePtr node;
      // IntArrayRef dims_ = {2, 3, 1, 0};
      int64_t dim_out_pos[] = {2, 3, 1, 0};
      int64_t dim_out_pos_3d[] = {2, 3, 4, 1, 0};
      IntArrayRef dims_ = dim_out_pos;
      if (self.dim() == 5)
        dims_ = dim_out_pos_3d;

      if (layout_format != habana_lazy::LayoutFormat::kHWCK) {
        node =
            std::make_shared<ir::Permute>(self, dims_, "hpu::permute_weight");
      } else {
        node = std::make_shared<ir::Permute>(
            self, dims_, "hpu::permuted_weight_restride");
      }
      result = empty_strided_hpu_lazy(
          self.sizes(), self.strides(), self.options(), false);
      auto hl_result = GetHbLazyTensor(result);
      ir::Value& out = hl_result.CurrentIrValue();
      out.SetNode(
          node,
          hl_result.GetDevice(),
          hl_result.GetSizes(),
          hl_result.dtype_optional());
      hb_tensor.SetTensorLayout(habana_lazy::LayoutFormat::kHWCK);
    }
  }
  return result;
}

Tensor convolution_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  PT_LAZY_TRACE;
  Tensor weight_hpu = weight;
  if (weight.device().type() == c10::DeviceType::CPU &&
      (habana_lazy::exec::OptPassCfg::GetInstance()
           ->IsEnabledWeightPermutePass()))
    weight_hpu = weight.to(c10::kHPU, true);
  auto weight_hwck = permute_wt_hpu(weight_hpu);
  bool is_weight_hwck = (habana_lazy::exec::OptPassCfg::GetInstance()
                             ->IsEnabledWeightPermutePass())
      ? false
      : true;
  if (weight_hwck.device().type() == c10::DeviceType::CPU &&
      (!habana_lazy::exec::OptPassCfg::GetInstance()
            ->IsEnabledWeightPermutePass())) {
    auto is_5d_layout = weight_hwck.dim() == 5;
    c10::MemoryFormat memory_format = is_5d_layout
        ? c10::MemoryFormat::ChannelsLast3d
        : c10::MemoryFormat::ChannelsLast;
    std::array<int64_t, 4> swapped_dims_4d = {2, 3, 1, 0};
    std::array<int64_t, 5> swapped_dims_5d = {2, 3, 4, 1, 0};
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
          is_weight_hwck)});
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
  LazyOp<at::Tensor> k{
      "aten::constant_pad_nd",
      {self, pad, value},
      {1, 2},
      {PadOperator::compute_output_shape(self, pad)}};
  return k.call();
}
Tensor embedding_hpu_lazy(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  PT_LAZY_TRACE;
  ir::NodePtr embedding_node = std::make_shared<ir::Embedding_forward>(
      weight, indices, padding_idx, scale_grad_by_freq, sparse);

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
  return op.call();
}
Tensor embedding_dense_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  PT_LAZY_TRACE;
  ir::NodePtr embedding_bwd_node = std::make_shared<ir::Embedding_backward>(
      grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  std::vector<int64_t> sizes{num_weights, grad.size(-1)};
  LazyOp<at::Tensor, ir::Embedding_backward> op(
      embedding_bwd_node,
      {grad, indices, num_weights, padding_idx, scale_grad_by_freq},
      {sizes});
  return op.call();
}
Tensor embedding_bag_sum_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_LAZY_TRACE;
  ir::NodePtr node = std::make_shared<ir::EmbeddingBagSum>(
      input, indices, offsets, valid_count, kernel_mode);
  std::vector<int64_t> sizes{offsets.sizes()[0] - 1, input.size(1)};
  LazyOp<at::Tensor, ir::EmbeddingBagSum> op(
      node, {input, indices, offsets, valid_count, kernel_mode}, {sizes});
  return op.call();
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

  return op.call();
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
  return op.call(out);
}
Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_lazy(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_LAZY_TRACE;
  ir::NodePtr node = std::make_shared<ir::EmbeddingBagSumBwd>(
      out, input, indices, offsets, valid_count, kernel_mode);
  LazyOp<at::Tensor&, ir::EmbeddingBagSumBwd> op(
      node, {out, input, indices, offsets, valid_count, kernel_mode});
  return op.call(out);
}

Tensor& fill_hpu_lazy_(Tensor& self, const Scalar& value) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::fill_", {self, value}};
  return k.call(self);
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
      "aten::_s_where",
      {mask, value, self},
      {},
      {},
      2 /*output metadata is picked from self*/);

  Tensor where_out = where_op.call();
  // add a control edge as we add a loop using d2d copy back to self
  auto hl_self = GetOrCreateHbLazyTensor(self);
  updateDstDependencies(hl_self, self, true);
  // Adding memcpy to copy the output back to self as this is an inplace op
  AddMemcpy(where_out, self);
  return self;
}

Tensor& masked_fill_scalar_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    const Scalar& value) {
  PT_LAZY_TRACE;
  Tensor value_tensor = habana_helpers::scalar_to_device_tensor(value, self, 0);
  return masked_fill_hpu_lazy_(self, mask, value_tensor);
}
Tensor gather_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad) {
  PT_LAZY_TRACE;

  if (self.dim() != index.dim()) {
    auto shape = GatherOperator::compute_output_shape(self, dim_, index);
    LazyOp<at::Tensor> k{
        "aten::gather", {self, dim_, index, sparse_grad}, {1, 3}, {shape}};
    return k.call();
  }

  Tensor valid_count_tensor;
  c10::optional<at::Tensor> valid_count =
      c10::make_optional(valid_count_tensor);
  // we don't support unsorted as of now. Hence, setting sorted to true
  auto shape = GatherOperator::compute_output_shape(self, dim_, index);
  LazyOp<at::Tensor> k{
      "hpu::gather_elements",
      {self, index, valid_count, dim_, true},
      {3, 4},
      {shape}};
  auto result = k.call();

  flush_op(result);
  return result;
}

Tensor scatter_add_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k{"aten::scatter_add", {self, dim_, index, src}};
  return k.call();
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
  updateDstDependencies(hl_self, self, true);
  // Create MemCopy operator to copy value into self
  AddMemcpy(result, self);
  return self;
}

Tensor index_hpu_lazy(const at::Tensor& self, at::TensorList indices_in) {
  PT_LAZY_TRACE;

  std::vector<Tensor> indices_vec_out{};

  std::vector<Tensor> indices_vec(indices_in.vec());
  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].device().type() != c10::DeviceType::HPU) {
      indices_vec[i] = indices_vec[i].to(c10::kHPU);
    }
  }

  // handle views for tensorlist indices
  TensorList indices_in_list(indices_vec);
  indices_vec = HandleViewsTensorList(indices_in_list);

  // for case where indices are Boolean tensor(s), convert these to integer
  // indices using nonzero operator before calling index
  if (indices_vec[0].scalar_type() == c10::ScalarType::Bool) {
    for (size_t i = 0; i < indices_vec.size(); i++) {
      auto list = torch::nonzero_numpy(indices_vec.at(i));
      indices_vec_out.insert(
          indices_vec_out.cend(), list.cbegin(), list.cend());
    }
  }
  at::TensorList indices =
      (indices_vec[0].scalar_type() == c10::ScalarType::Bool) ? indices_vec_out
                                                              : indices_vec;

  auto indices_out_vec = HandleViewsTensorList(indices);
  TensorList indices_out_list(indices_out_vec);

  // for this particular indices configuration gather_mxnet throws GC
  // compilation error, therefore use simple gather for now
  if (indices_out_list.size() == 1 && indices_out_list[0].dim() == 1) {
    auto shape =
        GatherOperator::compute_output_shape(self, 0, indices_out_list[0]);
    LazyOp<at::Tensor> k{
        "aten::gather", {self, 0, indices_out_list[0], false}, {1, 3}, {shape}};
    return k.call();
  }

  // additional casts inserted for handling dtypes other than f32/bf16 because
  // gather_mxnet TPC kernel used supports only f32/bf16
  at::Tensor self_cast = self;
  if (self.scalar_type() != c10::ScalarType::Double &&
      self.scalar_type() != c10::ScalarType::Float &&
      self.scalar_type() != c10::ScalarType::BFloat16) {
    // i8/i16/i32 -> f32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {self, c10::ScalarType::Float},
        {self.sizes().vec()},
        c10::ScalarType::Float};
    self_cast = k_.call();
  }

  LazyOp<at::Tensor> k{
      "aten::index",
      {self_cast, indices_out_list},
      {},
      {IndexOperator::compute_output_shape(self_cast, indices_out_list)}};
  auto result = k.call();

  if (self.scalar_type() != c10::ScalarType::Double &&
      self.scalar_type() != c10::ScalarType::Float &&
      self.scalar_type() != c10::ScalarType::BFloat16) {
    auto out_type = (self.scalar_type() == c10::ScalarType::Long)
        ? (c10::ScalarType::Int)
        : self.scalar_type();
    LazyOp<at::Tensor> k_{
        "hpu::cast", {result, out_type}, {result.sizes().vec()}, out_type};
    return k_.call();
  }

  return result;
}

Tensor& _index_put_impl_hpu_lazy_(
    Tensor& self,
    at::TensorList indices,
    const Tensor& value,
    bool accumulate,
    UNUSED const bool unsafe) {
  PT_LAZY_TRACE;
  // index backward is not supported on hpu, indices needs to be
  // bool, byte or long type for cpu fallback
  return index_put_hpu_lazy_(self, indices, value, accumulate);
}

Tensor nonzero_hpu_lazy(const Tensor& self) {
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
    return output;
  }

  // Add nonzero node
  std::vector<int64_t> output_shape{elements, dimensions};
  std::vector<int64_t> shape_tensor_shape{5};
  using T = std::tuple<at::Tensor, at::Tensor>;
  LazyOp<T> k("hpu::nonzero", {self}, {}, {output_shape, shape_tensor_shape});
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
  HbLazyTensor::SyncTensorsGraph(&hl_flush_end);
  auto cpu_end_tensor = end_tensor.to(c10::kCPU);
  auto end = cpu_end_tensor.item<int64_t>();

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
  auto sliced_shape = DimVector{end, dimensions};
  auto node = std::make_shared<ir::Slice>(where_tensor, 0, 0, end, 1);
  auto result = empty_hpu_lazy(
      sliced_shape, hb_options, self.suggest_memory_format(), true);
  auto hl_result = GetOrCreateHbLazyTensor(result, c10::kHPU);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
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

  // add a control edge as we add a loop using d2d copy back to self
  updateDstDependencies(hl_self, self, true);

  auto hl_index_add_out = GetHbLazyTensor(index_add_out);

  auto copy_node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::habana_d2d_memcpy_other"),
      {hl_index_add_out.GetIrValue(), hl_self.GetIrValue()});

  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      self.device().index());
  context->MarkTensorRegistered(hl_self.getTensorUniqueId());
  habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
  out.SetNode(
      copy_node,
      hl_self.GetDevice(),
      hl_self.GetSizes(),
      hl_self.dtype_optional());

  flush_op(self);
  return self;
}

Tensor index_put_hpu_lazy(
    const Tensor& self,
    TensorList indices_in,
    const Tensor& value_in,
    bool accumulate) {
  PT_LAZY_TRACE;
  std::vector<Tensor> indices_vec{indices_in.vec()};
  std::vector<Tensor> indices_vec_out{};
  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].device().type() != c10::DeviceType::HPU) {
      indices_vec[i] = indices_vec[i].to(c10::kHPU);
    }
  }

  // handle views for tensorlist indices
  TensorList indices_in_list(indices_vec);
  indices_vec = HandleViewsTensorList(indices_in_list);

  // for case where indices are Boolean tensor(s), convert these to integer
  // indices using nonzero operator before calling index
  if (indices_vec[0].scalar_type() == c10::ScalarType::Bool) {
    // do a mark_step to avoid attaching the select + scatter to a larger
    // previous graph
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

  auto indices_out_vec = HandleViewsTensorList(indices);
  TensorList indices_out_list(indices_out_vec);

  // Assuming if 1st indices tensor is ZST then other indices tensors in list
  // (if any) will be ZST too. For ZST indices tensor broadcast and scatter_nd
  // operations are throwing GC errors therefore we have this workaround to
  // return a copy of input tensor.
  // TBD: Investigate further and raise a JIRA on GC.
  if (indices_out_list[0].numel() == 0) {
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
      self.scalar_type() != c10::ScalarType::BFloat16) {
    // i8/i16/i32 -> f32
    LazyOp<at::Tensor> k_{
        "hpu::cast",
        {self, c10::ScalarType::Float},
        {self.sizes().vec()},
        c10::ScalarType::Float};
    self_cast = k_.call();
    // i8/i16/i32 -> f32
    LazyOp<at::Tensor> kv_{
        "hpu::cast",
        {value_in, c10::ScalarType::Float},
        {value_in.sizes().vec()},
        c10::ScalarType::Float};
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
        self.scalar_type() != c10::ScalarType::BFloat16) {
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
    for (size_t i = 0; i < indices_shape.size() - 1; i++)
      mul_factor_v.push_back(mul_factor_v[i] * indices_shape[i]);
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
        self.scalar_type() != c10::ScalarType::BFloat16) {
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

Tensor& index_put_hpu_lazy_(
    at::Tensor& self,
    TensorList indices_in,
    const at::Tensor& value,
    bool accumulate) {
  PT_LAZY_TRACE;
  std::vector<Tensor> indices_vec{indices_in.vec()};
  auto isIndicesBool = indices_vec[0].scalar_type() == c10::ScalarType::Bool;
  auto index_put_result =
      index_put_hpu_lazy(self, indices_in, value, accumulate);
  auto hl_self = GetOrCreateHbLazyTensor(self);
  // add a control edge as we add a loop using d2d copy back to self
  updateDstDependencies(hl_self, self, true);

  auto hl_index_put_out = GetHbLazyTensor(index_put_result);

  auto copy_node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::habana_d2d_memcpy_other"),
      {hl_index_put_out.GetIrValue(), hl_self.GetIrValue()});

  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      self.device().index());
  context->MarkTensorRegistered(hl_self.getTensorUniqueId());
  habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
  out.SetNode(
      copy_node,
      hl_self.GetDevice(),
      hl_self.GetSizes(),
      hl_self.dtype_optional());

  HandleViewsD2D(index_put_result, self);

  if (isIndicesBool) {
    std::vector<HbLazyTensor> hl_flush_end = {GetHbLazyTensor(self)};
    HbLazyTensor::SyncTensorsGraph(&hl_flush_end);
  } else {
    flush_op(self);
  }
  return self;
}

Tensor index_select_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index) {
  PT_LAZY_TRACE;
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto node = std::make_shared<ir::IndexSelect>(self, dim, index);

  auto shape = GatherOperator::compute_output_shape(self, dim, index);
  auto result = empty_hpu_lazy(
      shape, self.options(), self.suggest_memory_format(), false);

  auto hl_result = GetOrCreateHbLazyTensor(result, c10::kHPU);

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{self, index};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}
Tensor gather2d_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  PT_LAZY_TRACE;
  HABANA_ASSERT(0);
  return gather2d_hpu(input, indices, validCount);
}
Tensor slice_hpu_lazy(
    const Tensor& self_in,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  PT_LAZY_TRACE;
  if ((self_in.dim() <= 1) || (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VIEW_TABLE))) {
    return at::native::slice(self_in, dim, start, end, step);
  }
  // check if output tensor will be ZST
  auto sum_elm = habana_helpers::tensor_numel(self_in);
  auto out_zst = (!sum_elm) ||
      ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
       (start.value() == end.value()));

  // WA for https://jira.habana-labs.com/browse/SW-37197
  auto self = self_in;
  auto dim_orig = dim;
  if ((dim == self_in.dim() - 1) && (step > 1) && !out_zst) {
    self = transpose_hpu_lazy(self_in, self_in.dim() - 1, self_in.dim() - 2);
    dim = self.dim() - 2;
  }
  auto node =
      std::make_shared<ir::Slice>(self, dim, start.value(), end.value(), step);

  auto shape = SliceOperator::compute_output_shape(
      self, dim, start.value(), end.value(), step);

  // This ZST output tensor should ideally be handled at Synapse level, but
  // since it is throwing errors in that case we are forced to add this
  // work-around.
  // TBD: Investigate and raise a JIRA on GC. SW-57116 already raised in context
  // of transformer.
  if (out_zst) {
    auto result = empty_hpu_lazy(
        shape, self.options(), self.suggest_memory_format(), true);
    auto hl_result = GetHbLazyTensor(result);
    updateDstDependencies(hl_result, result);
    flush_op(result);
    return result;
  }
  auto result = empty_hpu_lazy(
      shape, self.options(), self.suggest_memory_format(), false);
  auto hl_result = GetHbLazyTensor(result);

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  auto output = result;
  // WA for https://jira.habana-labs.com/browse/SW-37197
  if ((dim_orig == self_in.dim() - 1) && (step > 1)) {
    output = transpose_hpu_lazy(result, result.dim() - 1, result.dim() - 2);
  }
  flush_op(output);
  return output;
}

Tensor slice_backward_hpu_lazy_legacy(
    const Tensor& self,
    const Tensor& grad_output,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  PT_LAZY_TRACE;
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  auto sizes = self.sizes().vec();
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }
  auto index_size = grad_output.sizes().vec();
  std::vector<int64_t> shape(grad_output.dim(), 1);
  shape[dim] = index_size[dim];
  auto index = empty_hpu_lazy(
      index_size[dim],
      grad_output.options().dtype(c10::ScalarType::Int),
      grad_output.suggest_memory_format(),
      false);
  auto grad_input = at::native::zeros_like(
      self,
      c10::optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt(),
      self.suggest_memory_format());
  index = at::native::arange(
      start,
      end,
      step,
      c10::optTypeMetaToScalarType(index.options().dtype_opt()),
      index.options().layout_opt(),
      index.options().device_opt(),
      index.options().pinned_memory_opt());
  auto expand_idx = index.reshape(IntArrayRef(shape)).expand(index_size);
  auto result = HpuOp::scatter(grad_input, dim, expand_idx, grad_output);

  return result;
}

Tensor slice_backward_hpu_lazy(
    const Tensor& self,
    const Tensor& grad_output,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  PT_LAZY_TRACE;
  Tensor result;

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VIEW_TABLE)) {
    auto ndim = self.dim();
    auto size = self.sizes();

    bool is_cl =
        ((self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d) ||
         (self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast));

    if (ndim == 0) {
      TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
    }
    dim = at::maybe_wrap_dim(dim, ndim);

    // always set contigous strides. This is because for multidimensional
    // slices, self.strides() will have swapped strides that is applicable only
    // for CPU handle chlast and chlast3d appropriately

    std::vector<int64_t> out_size_vec;
    c10::MemoryFormat mf;

    if (is_cl && (size.size() == 4)) {
      // NCHW -> NHWC
      const int64_t dim_pos_in[4] = {0, 2, 3, 1};
      for (size_t idx = 0; idx < size.size(); idx++) {
        out_size_vec.emplace_back(size[dim_pos_in[idx]]);
      }

      const int64_t dim_translate_pos[4] = {0, 3, 1, 2};
      dim = dim_translate_pos[dim];
      mf = c10::MemoryFormat::ChannelsLast;

    } else if (is_cl && (size.size() == 5)) {
      // NCDHW -> NDHWC
      const int64_t dim_pos_in[5] = {0, 2, 3, 4, 1};
      for (size_t idx = 0; idx < size.size(); idx++) {
        out_size_vec.emplace_back(size[dim_pos_in[idx]]);
      }

      const int64_t dim_translate_pos[5] = {0, 4, 1, 2, 3};
      dim = dim_translate_pos[dim];
      mf = c10::MemoryFormat::ChannelsLast3d;

    } else {
      for (size_t idx = 0; idx < size.size(); idx++) {
        out_size_vec.emplace_back(size[idx]);
      }
    }

    std::vector<int64_t> strides_vec(out_size_vec.size(), 1);
    for (auto i = out_size_vec.size(); i > 1; --i) {
      strides_vec[i - 2] = strides_vec[i - 1] * out_size_vec[i - 1];
    }

    DimVector strides(strides_vec);
    DimVector sizes(out_size_vec);

    // TODO: support negative strides
    TORCH_CHECK(step > 0, "slice step must be positive");

    // INT64_MAX stands for default value.
    if (start == INT64_MAX) {
      start = 0;
    }
    if (start < 0) {
      start += sizes[dim];
    }
    if (end < 0) {
      end += sizes[dim];
    }
    if (start < 0) {
      start = 0;
    } else if (start >= sizes[dim]) {
      start = sizes[dim];
    }
    if (end < start) {
      end = start;
    } else if (end >= sizes[dim]) {
      end = sizes[dim];
    }

    // auto storage_offset = self.storage_offset() + start * strides[dim];
    auto storage_offset = start * strides[dim];
    auto len = end - start;
    sizes[dim] = (len + step - 1) / step; // round-up
    strides[dim] *= step;

    // convert c10::List to IntArrayRef
    std::vector<int64_t> size_vec;
    for (size_t idx = 0; idx < size.size(); idx++) {
      size_vec.emplace_back(size[idx]);
    }

    IntArrayRef grad_in_size(size_vec);

    auto grad_input =
        empty_hpu_lazy(grad_in_size, grad_output.options(), mf, true);
    grad_input = zero_hpu_lazy(grad_input);

    result = add_strided_insert_node(
        grad_input, grad_output, strides, storage_offset);
  } else {
    result = slice_backward_hpu_lazy_legacy(
        self, grad_output, dim, start, end, step);
  }

  return result;
};

Tensor select_hpu_lazy(const Tensor& self, int64_t dim, int64_t index) {
  PT_LAZY_TRACE;

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_VIEW_TABLE)) {
    return at::native::select(self, dim, index);
  }

  int64_t ndim = self.dim();
  if (ndim == 0) {
    HABANA_ASSERT(false, "select() cannot be applied to a 0-dim tensor.")
  }
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    if (self.has_names() && self.names()[dim] != Dimname::wildcard()) {
      HABANA_ASSERT(
          false,
          "select(): index ",
          index,
          " out of range for tensor of size ",
          self.sizes(),
          " at dimension ",
          self.names()[dim]);
    }
    HABANA_ASSERT(
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

  auto node = std::make_shared<ir::Slice>(self, dim, index);

  // infer shape
  auto shape = SelectOperator::compute_output_shape(self, dim);
  auto result = empty_hpu_lazy(
      shape, self.options(), self.suggest_memory_format(), false);
  auto hl_result = GetHbLazyTensor(result);

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor& arange_hpu_lazy(
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
  std::vector<at::Tensor> input_pt_vec{output};

  // Currently synapse support dynamic shape arange only for int datatypes.
  // For any other output datatype, will fallback to normal flow.
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES) &&
      (output.scalar_type() == c10::ScalarType::Int)) {
    std::vector<int64_t> params_vec{step.toInt(), end.toInt(), start.toInt()};
    auto input_size = IntArrayRef(params_vec.data(), params_vec.size());
    auto params_shape = empty_hpu_lazy(
        input_size,
        output.options(),
        output.suggest_memory_format(),
        false,
        INPUT_DESCRIBING_SHAPE_TENSOR);
    auto hl_params_shape = GetOrCreateHbLazyTensor(params_shape, c10::kHPU);
    node = ir::Node::Create(
        Symbol::fromQualString("hpu::arange_out_ds"),
        {hl_params_shape.GetIrValue(), hl_result.GetIrValue()});
    input_pt_vec.emplace_back(params_shape);
  } else {
    auto hl_start = GetIrValueForScalar(start);
    auto hl_end = GetIrValueForScalar(end);
    auto hl_step = GetIrValueForScalar(step);
    node = ir::Node::Create(
        Symbol::fromQualString("hpu::arange_out"),
        {hl_start, hl_end, hl_step, hl_result.GetIrValue()});
  }

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  // updatet the view if any
  updateDstDependencies(hl_result, output);

  node->AddInputPtTensors(input_pt_vec);
  flush_op(output);
  return output;
}

Tensor mm_hpu_lazy(const at::Tensor& mat1, const at::Tensor& mat2) {
  PT_LAZY_TRACE;
  LazyOp<Tensor> k{
      "aten::mm",
      {mat1, mat2},
      {},
      {MMOperator::compute_output_shape(mat1, mat2)}};
  return k.call();
}

Tensor addmm_hpu_lazy(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  PT_LAZY_TRACE;
  const std::vector<int64_t> shape_out = {mat1.size(0), mat2.size(1)};
  LazyOp<Tensor> k{
      "aten::addmm", {self, mat1, mat2, beta, alpha}, {}, {shape_out}};
  return k.call();
}

Tensor& batch_gemm_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& mat2) {
  PT_LAZY_TRACE;
  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
  auto hl_mat2 = GetOrCreateHbLazyTensor(mat2, c10::kHPU);

  hl_self = HandleViewsOrUpdate(self, hl_self);
  hl_mat2 = HandleViewsOrUpdate(mat2, hl_mat2);

  const auto node = ir::Node::Create(
      Symbol::fromQualString("aten::bmm"),
      {hl_self.GetIrValue(), hl_mat2.GetIrValue()});

  auto hlresult = GetHbLazyTensor(out);
  ir::Value& out_val = hlresult.CurrentIrValue();
  out_val.SetNode(
      node,
      hlresult.GetDevice(),
      hlresult.GetSizes(),
      hlresult.dtype_optional());
  // updatet the view if any
  updateDstDependencies(hlresult, out);
  std::vector<at::Tensor> input_pt_vec{self, mat2};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(out);
  return out;
}

Tensor batch_gemm_hpu_lazy(const Tensor& self, const Tensor& mat2) {
  PT_LAZY_TRACE;
  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
  auto hl_mat2 = GetOrCreateHbLazyTensor(mat2, c10::kHPU);

  hl_self = HandleViewsOrUpdate(self, hl_self);
  hl_mat2 = HandleViewsOrUpdate(mat2, hl_mat2);

  const auto node = ir::Node::Create(
      Symbol::fromQualString("aten::bmm"),
      {hl_self.GetIrValue(), hl_mat2.GetIrValue()});

  auto shape_out = BmmOperator::compute_output_shape(self, mat2);
  const auto result = empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  const auto hlresult = GetHbLazyTensor(result);
  ir::Value& out = hlresult.CurrentIrValue();
  out.SetNode(
      node,
      hlresult.GetDevice(),
      hlresult.GetSizes(),
      hlresult.dtype_optional());
  updateDstDependencies((HbLazyTensor&)hlresult, (Tensor&)result);
  std::vector<at::Tensor> input_pt_vec{self, mat2};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor dot_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  // TODO: Need to make 0-dim output.
  // With ... shape_out = {0}; following error reported
  // Assertion (dst.nbytes() >= src.nbytes()) is false!
  // ... /pytorch-integration/habana_kernels/basic_kernels.cpp:147 copy_hpu_
  // terminate called without an active exception
  // Aborted (core dumped)
  std::vector<int64_t> shape_out = {1};
  LazyOp<at::Tensor> k{"aten::dot", {self, other}, {}, {shape_out}};
  return k.call();
}

Tensor mv_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  std::vector<int64_t> shape_out = {self.size(0)};
  LazyOp<at::Tensor> k{"aten::mv", {self, other}, {}, {shape_out}};
  return k.call();
}

std::tuple<Tensor, Tensor> nll_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  PT_LAZY_TRACE;

  std::vector<int64_t> out_shape;
  if (reduction ==
      at::Reduction::Reduction::None) { // consider input in nchw format
    out_shape.emplace_back(self.sizes()[0]);
  } else {
    out_shape.emplace_back(1);
  }
  using T = std::tuple<at::Tensor, at::Tensor>;
  LazyOp<T> k(
      "aten::nll_loss_forward",
      {self, target, weight, reduction, ignore_index},
      {3, 4}, // metadata_indices
      {out_shape, {}} // out_shapes
  );
  return k.call();
}

std::tuple<Tensor, Tensor> nll_loss2d_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  PT_LAZY_TRACE;

  std::vector<int64_t> out_shape;
  if (reduction ==
      at::Reduction::Reduction::None) { // consider input in nchw format
    out_shape.emplace_back(self.sizes()[0]);
    out_shape.emplace_back(self.sizes()[2]);
    out_shape.emplace_back(self.sizes()[3]);
  } else {
    out_shape.emplace_back(1);
  }
  using T = std::tuple<at::Tensor, at::Tensor>;
  LazyOp<T> k(
      "aten::nll_loss2d_forward",
      {self, target, weight, reduction, ignore_index},
      {}, // metadata_indices
      {out_shape, {}} // out_shapes
  );

  T results = k.call();
  Tensor output = std::get<0>(results);
  if (reduction != at::Reduction::Reduction::None) {
    output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  return std::make_tuple(output, std::get<1>(results));
}

Tensor nll_loss2d_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k(
      "aten::nll_loss2d_backward",
      {grad_output,
       self,
       target,
       weight,
       reduction,
       ignore_index,
       total_weight},
      {}, /* metadata_indices */
      {self.sizes().vec()} /* out_shapes*/);

  return k.call();
}

Tensor nll_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k(
      "aten::nll_loss_backward",
      {grad_output,
       self,
       target,
       weight,
       reduction,
       ignore_index,
       total_weight},
      {4, 5}, /* metadata_indices */
      {self.sizes().vec()} /* out_shapes*/);

  return k.call();
}

Tensor mse_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k(
      "aten::mse_loss",
      {self, target, reduction},
      {2}, // metadata_indices
      {MSELossFwdOperator::compute_output_shape(self, reduction)});
  return k.call();
}

Tensor mse_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  PT_LAZY_TRACE;
  auto shape = MSELossBwdOperator::compute_output_shape(self);
  LazyOp<at::Tensor> k(
      "aten::mse_loss_backward",
      {grad_output, self, target, reduction},
      {3}, // metadata_indices
      {shape});
  return k.call();
}

Tensor binary_cross_entropy_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  PT_LAZY_TRACE;
  ir::NodePtr bce_loss_node =
      std::make_shared<ir::BceLoss_forward>(self, target, weight, reduction);
  auto sizes = BceFwdOperator::compute_output_shape(self, reduction);
  LazyOp<at::Tensor, ir::BceLoss_forward> k{
      bce_loss_node, {self, target, weight, reduction}, {sizes}};
  return k.call();
}

Tensor binary_cross_entropy_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  PT_LAZY_TRACE;
  ir::NodePtr bce_bwd_loss_node = std::make_shared<ir::BceLoss_backward>(
      grad_output, self, target, weight, reduction);
  LazyOp<at::Tensor, ir::BceLoss_backward> k{
      bce_bwd_loss_node,
      {grad_output, self, target, weight, reduction},
      {self.sizes().vec()}};
  return k.call();
}

Tensor binary_cross_entropy_with_logits_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& pos_weight,
    int64_t reduction) {
  PT_LAZY_TRACE;
  std::vector<int64_t> sizes = {1};
  if (reduction == at::Reduction::Reduction::None) {
    sizes = self.sizes().vec();
  }

  LazyOp<at::Tensor> k(
      "aten::binary_cross_entropy_with_logits",
      {self, target, pos_weight, weight, reduction},
      {4},
      {sizes});
  return k.call();
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
  return k.call();
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
  return k.call();
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean_,
    const Tensor& running_var_,
    bool training,
    double momentum,
    double eps) {
  PT_LAZY_TRACE;
  ir::NodePtr node;

  Tensor running_mean, running_var, residual_add;
  // if RMV are undefined, create zero mean and unit variance tensors for
  // numerical stability of BN. Note that they should have same
  // dtype as weight

  if (!running_mean_.defined()) {
    IntArrayRef rm_size;
    if (input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast) {
      rm_size = input.sizes()[3];
    } else if (
        input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d) {
      rm_size = input.sizes()[4];
    } else {
      rm_size = input.sizes()[1];
    }

    running_mean = empty_hpu_lazy(
        rm_size, weight.options(), input.suggest_memory_format(), true);
    fill_hpu_lazy_(running_mean, 0);
  } else {
    running_mean = running_mean_;
  }

  if (!running_var_.defined()) {
    IntArrayRef rv_size;
    if (input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast) {
      rv_size = input.sizes()[3];
    } else if (
        input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d) {
      rv_size = input.sizes()[4];
    } else {
      rv_size = input.sizes()[1];
    }
    running_var = empty_hpu_lazy(
        rv_size, weight.options(), input.suggest_memory_format(), true);
    fill_hpu_lazy_(running_var, 1);
  } else {
    running_var = running_var_;
  }

  if (training) {
    residual_add = empty_hpu_lazy(
        {1}, input.options(), input.suggest_memory_format(), true);
    node = std::make_shared<ir::BatchNormForward>(
        input,
        weight,
        bias,
        residual_add,
        running_mean,
        running_var,
        training,
        momentum,
        eps);
  } else {
    // weight and bias positions are swapped to match TPC kernel signature
    node = std::make_shared<ir::BatchNormInf>(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps);
  }

  auto output_sizes = input.sizes().vec();
  auto sizes = std::make_tuple(
      output_sizes, running_mean.sizes().vec(), running_var.sizes().vec());
  auto mf = input.suggest_memory_format();

  // Get Output Image
  auto result_img =
      empty_hpu_lazy(std::get<0>(sizes), input.options(), mf, false);
  const auto hlresult0 = GetHbLazyTensor(result_img);
  ir::Value& out0 = hlresult0.CurrentIrValue();
  out0.SetNode(
      node,
      hlresult0.GetDevice(),
      hlresult0.GetSizes(),
      hlresult0.dtype_optional());

  // set the running mean and variance as output nodes
  if (training) {
    const auto hlresult1 = GetHbLazyTensor(running_mean);
    ir::Value& out1 = hlresult1.CurrentIrValue();
    out1.SetNode(
        node,
        hlresult1.GetDevice(),
        hlresult1.GetSizes(),
        hlresult1.dtype_optional(),
        1);

    const auto hlresult2 = GetHbLazyTensor(running_var);
    ir::Value& out2 = hlresult2.CurrentIrValue();
    out2.SetNode(
        node,
        hlresult2.GetDevice(),
        hlresult2.GetSizes(),
        hlresult2.dtype_optional(),
        2);
  }

  Tensor result_mean, result_var;
  if (training) {
    // Get output mean and var
    result_mean =
        empty_hpu_lazy(std::get<1>(sizes), running_mean.options(), mf, false);
    const auto hlresult3 = GetHbLazyTensor(result_mean);
    ir::Value& out3 = hlresult3.CurrentIrValue();
    out3.SetNode(
        node,
        hlresult3.GetDevice(),
        hlresult3.GetSizes(),
        hlresult3.dtype_optional(),
        3);

    result_var =
        empty_hpu_lazy(std::get<2>(sizes), running_var.options(), mf, false);
    const auto hlresult4 = GetHbLazyTensor(result_var);
    ir::Value& out4 = hlresult4.CurrentIrValue();
    out4.SetNode(
        node,
        hlresult4.GetDevice(),
        hlresult4.GetSizes(),
        hlresult4.dtype_optional(),
        4);
  }

  if (training) {
    flush_op({result_img, result_mean, result_var, running_mean, running_var});
    return std::make_tuple(result_img, result_mean, result_var);
  } else {
    flush_op({result_img, running_mean, running_var});
    return std::make_tuple(result_img, running_mean, running_var);
  }
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd_hpu_lazy(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& running_mean_,
    const Tensor& running_var_,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    UNUSED std::array<bool, 3> output_mask) {
  PT_LAZY_TRACE;

  Tensor running_mean, running_var;
  // create tensors if RMV are undefined. Note that they should have same
  // dtype as weight

  if (!running_mean_.defined()) {
    IntArrayRef rm_size;
    if (input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast) {
      rm_size = input.sizes()[3];
    } else if (
        input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d) {
      rm_size = input.sizes()[4];
    } else {
      rm_size = input.sizes()[1];
    }

    running_mean = empty_hpu_lazy(
        rm_size, weight.options(), input.suggest_memory_format(), true);
  } else {
    running_mean = running_mean_;
  }

  if (!running_var_.defined()) {
    IntArrayRef rv_size;
    if (input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast) {
      rv_size = input.sizes()[3];
    } else if (
        input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d) {
      rv_size = input.sizes()[4];
    } else {
      rv_size = input.sizes()[1];
    }
    running_var = empty_hpu_lazy(
        rv_size, weight.options(), input.suggest_memory_format(), true);
  } else {
    running_var = running_var_;
  }

  ir::NodePtr node = std::make_shared<ir::BatchNormBackward>(
      grad_out,
      input,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_invstd,
      train,
      eps,
      output_mask);

  auto output_sizes = input.sizes().vec();
  auto sizes = std::make_tuple(
      output_sizes, running_mean.sizes().vec(), running_var.sizes().vec());

  at::Tensor result_1, result_2, result_3;
  if (output_mask[0]) {
    result_1 = empty_hpu_lazy(
        std::get<0>(sizes),
        input.options(),
        input.suggest_memory_format(),
        false);
    const auto hlresult_1 = GetHbLazyTensor(result_1);
    ir::Value& out_1 = hlresult_1.CurrentIrValue();
    out_1.SetNode(
        node,
        hlresult_1.GetDevice(),
        hlresult_1.GetSizes(),
        hlresult_1.dtype_optional());
  }

  if (output_mask[1]) {
    result_2 = empty_hpu_lazy(
        std::get<1>(sizes),
        weight.options(),
        weight.suggest_memory_format(),
        false);
    const auto hlresult_2 = GetHbLazyTensor(result_2);
    ir::Value& out_2 = hlresult_2.CurrentIrValue();
    out_2.SetNode(
        node,
        hlresult_2.GetDevice(),
        hlresult_2.GetSizes(),
        hlresult_2.dtype_optional(),
        1);
  }

  if (output_mask[2]) {
    result_3 = empty_hpu_lazy(
        std::get<2>(sizes),
        save_mean.options(),
        save_mean.suggest_memory_format(),
        false);
    const auto hlresult_3 = GetHbLazyTensor(result_3);
    ir::Value& out_3 = hlresult_3.CurrentIrValue();
    out_3.SetNode(
        node,
        hlresult_3.GetDevice(),
        hlresult_3.GetSizes(),
        hlresult_3.dtype_optional(),
        2);
  }

  flush_op({result_1, result_2, result_3});
  return std::make_tuple(result_1, result_2, result_3);
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_hpu_lazy(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    double eps) {
  PT_LAZY_TRACE;
  auto weight = weight_opt.value_or(Tensor());
  auto bias = bias_opt.value_or(Tensor());
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  const int normalized_ndim = normalized_shape.size();
  const int axis = input_ndim - normalized_ndim;
  const int64_t m =
      multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  // const int64_t n =
  //     multiply_integers(input_shape.cbegin() + axis, input_shape.cend());
  ir::NodePtr node = std::make_shared<ir::LayerNormForward>(
      input, normalized_shape, weight_opt, bias_opt, eps);

  auto sizes = LayerNormOperator::getOutputSizes(input, m);
  LazyOp<std::tuple<Tensor, Tensor, Tensor>, ir::LayerNormForward> k{
      node, {input, normalized_shape, weight_opt, bias_opt, eps}, sizes};
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
        : LazyOp<T, U>(std::move(node), {}, {}, -1),
          dY{std::move(dY)},
          X{std::move(X)},
          normalized_shape{std::move(normalized_shape)},
          mean{std::move(mean)},
          rstd{std::move(rstd)},
          weight_opt{std::move(weight_opt)},
          bias_opt{std::move(bias_opt)},
          grad_input_mask{std::move(grad_input_mask)} {}

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
    const at::Tensor& X;
    IntArrayRef normalized_shape;
    const at::Tensor& mean;
    const at::Tensor& rstd;
    const c10::optional<Tensor>& weight_opt;
    const c10::optional<Tensor>& bias_opt;
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

Tensor norm_scalar_hpu_lazy(const Tensor& self, const Scalar& p) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{
      "aten::norm", {self, p}, {}, {NormOperator::compute_output_shape()}};
  Tensor out = k.call();
  out.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  return out;
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
        : LazyOp<T, U>(std::move(node), {}, {}, -1),
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

      // return always nhwc. convert to nchw
      std::vector<long int> shape_out = {
          opsize_nhwc.at(0),
          opsize_nhwc.at(3),
          opsize_nhwc.at(1),
          opsize_nhwc.at(2)};
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
  return AtenHpuTypeDefault::max_pool2d_with_indices_backward_out(
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices,
      grad_input);
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

  // retunr always nhwc. convert to nchw
  std::vector<long int> out_shape = {
      opsize_nhwc.at(0),
      opsize_nhwc.at(3),
      opsize_nhwc.at(1),
      opsize_nhwc.at(2)};

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
      {input.sizes().vec()}};
  return k.call();
}

Tensor avg_pool2d_hpu_lazy(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  PT_LAZY_TRACE;
  ir::NodePtr avgpool_node = std::make_shared<ir::AvgPool>(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);

  // shape inference
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, false);

  // return always nhwc. convert to nchw
  std::vector<long int> shape_out = {
      opsize_nhwc.at(0),
      opsize_nhwc.at(3),
      opsize_nhwc.at(1),
      opsize_nhwc.at(2)};

  LazyOp<at::Tensor, ir::AvgPool> k{
      avgpool_node,
      {input,
       kernel_size,
       stride,
       padding,
       ceil_mode,
       count_include_pad,
       divisor_override},
      {shape_out}};
  return k.call();
}

Tensor& avg_pool2d_backward_out_hpu_lazy(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  PT_LAZY_TRACE;
  return AtenHpuTypeDefault::avg_pool2d_backward_out(
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      grad_input);
}
Tensor avg_pool2d_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  PT_LAZY_TRACE;
  ir::NodePtr avgpool_bwd_node = std::make_shared<ir::AvgPoolBackWard>(
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);

  // shape inference
  std::vector<int64_t> d{1, 1}; // setting dilation to 1 for avg pool
  IntArrayRef dilation(d.data(), d.size());
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, false);

  // retunr always nhwc. convert to nchw
  std::vector<long int> out_shape = {
      opsize_nhwc.at(0),
      opsize_nhwc.at(3),
      opsize_nhwc.at(1),
      opsize_nhwc.at(2)};

  TORCH_CHECK(grad_output.sizes().vec() == out_shape);
  LazyOp<at::Tensor, ir::AvgPoolBackWard> k{
      avgpool_bwd_node,
      {grad_output,
       input,
       kernel_size,
       stride,
       padding,
       ceil_mode,
       count_include_pad,
       divisor_override},
      {input.sizes().vec()}};
  return k.call();
}

Tensor adaptive_avg_pool2d_hpu_lazy(
    const Tensor& input,
    IntArrayRef output_size) {
  PT_LAZY_TRACE;
  auto opsize_nhwc =
      PoolHelper::compute_output_shape(input, output_size, false);
  // compute_output_shape return always nhwc. convert to nchw
  std::vector<long int> shape_out = {
      opsize_nhwc.at(0),
      opsize_nhwc.at(3),
      opsize_nhwc.at(1),
      opsize_nhwc.at(2)};

  LazyOp<Tensor> k{
      "aten::_adaptive_avg_pool2d", {input, output_size}, {1}, {shape_out}};
  return k.call();
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
  return k.call();
}

Tensor& randperm_hpu_lazy(
    Tensor& output,
    int64_t n,
    c10::optional<Generator> gen) {
  PT_LAZY_TRACE;

  // resizing the output as it is coming as empty from model
  auto hl_result = GetOrCreateHbLazyTensor(output, c10::kHPU);
  auto out_shape = DimVector({n});
  auto out_reshaped = hl_result.getAttachedTensorImpl();
  THHTensor_resizeNd(out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  output.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));

  LazyOp<Tensor&> op{
      "hpu::randperm_out",
      {Scalar((int32_t)n), std::move(gen), output},
      {1},
      {{n}}};

  return op.call(output);
}

std::tuple<Tensor, Tensor> fused_dropout_hpu_lazy(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen) {
  PT_LAZY_TRACE;
  struct FusedDropout : LazyOp<std::tuple<Tensor, Tensor>> {
    FusedDropout(const Tensor& self, double p, c10::optional<Generator> gen)
        : LazyOp<std::tuple<Tensor, Tensor>>(
              "aten::_fused_dropout",
              {self, p, std::move(gen)},
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

  FusedDropout op(self, p, std::move(gen));
  return op.call();
}

at::Tensor repeat_hpu_lazy(const at::Tensor& self, at::IntArrayRef repeats) {
  PT_LAZY_TRACE;
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
  return k.call();
}

Tensor sum_dim_IntList_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  at::Tensor self_cast = self;

  if (self.scalar_type() == c10::ScalarType::Bool ||
      self.scalar_type() == c10::ScalarType::Byte) {
    auto node = std::make_shared<ir::Cast>(self, c10::ScalarType::Int, true);
    at::TensorOptions hb_options = self.options().dtype(c10::ScalarType::Int);
    self_cast = empty_hpu_lazy(
        self.sizes(), hb_options, self.suggest_memory_format(), false);
    auto hl_cast = GetHbLazyTensor(self_cast);
    hl_cast = HandleViewsOrUpdate(self_cast, hl_cast);
    ir::Value& out = hl_cast.CurrentIrValue();
    out.SetNode(
        node,
        hl_cast.GetDevice(),
        hl_cast.GetSizes(),
        hl_cast.dtype_optional());
  }
  auto hl_self = GetOrCreateHbLazyTensor(self_cast, c10::kHPU);
  hl_self = HandleViewsOrUpdate(self_cast, hl_self);
  ir::NodePtr node =
      std::make_shared<ir::SumDimIntList>(self_cast, dim, keepdim, dtype);
  auto result = empty_hpu_lazy(
      ReduceOperator::compute_output_shape(self_cast, dim, keepdim),
      self_cast.options(),
      self_cast.suggest_memory_format(),
      false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor& sum_out_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype,
    Tensor& out) {
  PT_LAZY_TRACE;
  at::Tensor self_updated_dtype = self;

  if (dtype.has_value() && (dtype.value() != self.scalar_type())) {
    self_updated_dtype = self.to(dtype.value());
  }
  LazyOp<at::Tensor&> k(
      "aten::sum",
      {self_updated_dtype, dim, keepdim, dtype, out},
      {},
      {ReduceOperator::compute_output_shape(self_updated_dtype, dim, keepdim)});

  return k.call(out);
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
Tensor& mean_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  return AtenHpuTypeDefault::mean_out(self, dim, keepdim, dtype, output);
}

Tensor sum_hpu_lazy(const Tensor& self_in, c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  auto self = self_in;
  // Cast Boolean/Char (I8) inputs to Float since TPC kernel supports only f32
  if (self_in.scalar_type() == c10::ScalarType::Bool ||
      self_in.scalar_type() == c10::ScalarType::Char) {
    c10::ScalarType dst_dtype = c10::ScalarType::Float;
    self = empty_hpu_lazy(
        self_in.sizes(),
        self_in.options().dtype(dst_dtype),
        self_in.suggest_memory_format(),
        false);
    self = copy_hpu_lazy_(self, self_in, true);
  }

  ir::NodePtr node = std::make_shared<ir::Sum>(self, dtype);
  auto result =
      empty_hpu_lazy({0}, self.options(), self.suggest_memory_format(), false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  result.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  return result;
}

Tensor mean_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::mean", {self, dtype}, {}, {{1}}};
  return k.call();
}

Tensor prod_dim_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
  ir::NodePtr node =
      std::make_shared<ir::ProdDimInt>(self, dim, keepdim, dtype);

  auto result = empty_hpu_lazy(
      ReduceOperator::compute_output_shape(self, dim, keepdim),
      self.options(),
      self.suggest_memory_format(),
      false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor prod_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  ir::NodePtr node = std::make_shared<ir::Prod>(self, dtype);

  // Output of Prod is product of all elements
  std::vector<int64_t> shape_out{1};
  auto result = empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor& any_dim_out_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& output) {
  PT_LAZY_TRACE;

  std::vector<int64_t> shape_out =
      ReduceOperator::compute_output_shape(self, dim, keepdim);

  LazyOp<Tensor&> k{"aten::any", {self, dim, keepdim, output}, {}, {shape_out}};

  return k.call(output);
}

Tensor any_dim_hpu_lazy(const Tensor& self, int64_t dim, bool keepdim) {
  PT_LAZY_TRACE;

  struct Kernel : public LazyOp<at::Tensor> {
    explicit Kernel(const Tensor& self, int64_t dim, bool keepdim)
        : LazyOp<at::Tensor>("aten::any", {self, dim, keepdim}, {}, {}, -1),
          self(self),
          dim(dim),
          keepdim(keepdim) {}
    at::Tensor get_result_overrideable() override {
      return empty_hpu_lazy(
          ReduceOperator::compute_output_shape(self, dim, keepdim),
          self.options().dtype(c10::ScalarType::Bool),
          self.suggest_memory_format(),
          false);
    }
    at::Tensor self;
    int64_t dim;
    bool keepdim;
  };
  Kernel kernel{self, dim, keepdim};
  return kernel.call();
}

Tensor any_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  struct Kernel : public LazyOp<at::Tensor> {
    explicit Kernel(const Tensor& self)
        : LazyOp<at::Tensor>("aten::any", {self}, {}, {}, -1), self(self) {}
    at::Tensor get_result_overrideable() override {
      std::vector<int64_t> shape_out{1};

      return empty_hpu_lazy(
          shape_out,
          self.options().dtype(c10::ScalarType::Bool),
          self.suggest_memory_format(),
          false);
    }
    at::Tensor self;
  };
  Kernel kernel{self};
  return kernel.call();
}

Tensor argmax_hpu_lazy(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim) {
  PT_LAZY_TRACE;
  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
  ir::NodePtr node = std::make_shared<ir::ArgMax>(self, dim, keepdim);
  std::vector<int64_t> shape_out;

  if (dim.has_value()) {
    shape_out =
        ReduceOperator::compute_output_shape(self, dim.value(), keepdim);
  } else {
    shape_out.push_back(1);
  }

  auto result = empty_hpu_lazy(
      shape_out,
      self.options().dtype(c10::ScalarType::Int),
      self.suggest_memory_format(),
      false);
  auto hl_result = GetHbLazyTensor(result);

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor softmax_hpu_lazy(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<ir::LogSoftMax>(
      self, dim, half_to_float, "aten::_softmax");
  // infer shape
  auto shape_out = SoftmaxOperator::compute_output_shape(self);
  at::Tensor result;
  if (self.dtype() == c10::ScalarType::BFloat16) {
    result = empty_hpu_lazy(
        shape_out,
        self.options().dtype(c10::ScalarType::BFloat16),
        self.suggest_memory_format(),
        false);
  } else {
    result = empty_hpu_lazy(
        shape_out,
        self.options().dtype(c10::ScalarType::Float),
        self.suggest_memory_format(),
        false);
  }

  auto hl_result = GetHbLazyTensor(result);

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor softmax_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<ir::LogSoftMaxBackward>(
      grad, output, dim, input, "aten::_softmax_backward_data");
  // infer output shape
  auto shape_out = SoftmaxBackwardOperator::compute_output_shape(input);
  auto result = empty_hpu_lazy(
      shape_out, input.options(), input.suggest_memory_format(), false);

  auto hl_result = GetHbLazyTensor(result);

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor empty_hpu_lazy(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format,
    bool create_storage,
    synTensorType tensor_type) {
  PT_LAZY_TRACE;
  c10::optional<MemoryFormat> mem_format = optional_memory_format.has_value()
      ? optional_memory_format
      : options.memory_format_opt();
  auto original_dtype = options.dtype();
  auto type = typeMetaToScalarType(original_dtype);
  auto shape_tensor = habana_helpers::is_shape_tensor(tensor_type);
  if (!habana_helpers::is_supported_type(type)) {
    HABANA_ASSERT(shape_tensor == false);
    auto layout = options.layout();
    auto pinned_mem = options.pinned_memory();
    auto dev = c10::DeviceType::CPU;
    return at::empty(
        size, type, layout, dev, pinned_mem, optional_memory_format);
  }
  // Dont allocate 8 bytes for double/long as we are anyway going to cast at
  // CPU and then copy to device @ 4byts per element
  type = type == c10::ScalarType::Long ? c10::ScalarType::Int : type;
  type = type == c10::ScalarType::Double ? c10::ScalarType::Float : type;
  auto new_dtype = scalarTypeToTypeMeta(type);

  if (create_storage || shape_tensor) {
    c10 ::Allocator* allocator;
    if (options.pinned_memory()) {
      TORCH_CHECK(false, "habana allocator doesn't supported pinned memory");
    } else {
      allocator = habana::getHABANADeviceAllocator();
    }
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
    Tensor at_internal_tensor =
        AtenInternalHbTensor(std::move(storage_impl), new_dtype);
    // Setup the tensor sizes & strides for tensor with dim = 4, else for now
    // assuming contiguous
    if (tensor_type == DEVICE_SHAPE_TENSOR) {
      at_internal_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(
          device_shape_tensor_size);
    } else if ((4 == size.size()) && mem_format.has_value()) {
      at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          size, CalculateStrides(size, mem_format.value()));
    } else if ((5 == size.size()) && mem_format.has_value()) {
      at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          size, CalculateStrides5d(size, mem_format.value()));
    } else {
      at_internal_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
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
    auto context = habana_lazy_executor.getDeviceExecutionContext(
        options.device().index());
    if (context != nullptr) {
      auto exec_mode = context->getExecutionMode();
      is_in_lowering_mode = exec_mode == kLOWERING ? true : is_in_lowering_mode;
    }

    // This call could have come from a .to call and not from a lowering
    // context. In such case, create the lazt tensor.
    if (!is_in_lowering_mode) {
      HbLazyTensor hb_tensor = HbLazyTensor::CreateHbLazyTensor(
          size, 0, options.device(), typeMetaToScalarType(original_dtype));
      at_tensor = AtenFromHbLazyTensor(hb_tensor);

      // The lazy tensor will have a reference to the internal tensor
      hb_tensor.SetTensorData(at_internal_tensor);

      // Setup the tensor sizes & strides for tensor with dim = 4, else for
      // now assuming contiguous
      if (tensor_type == DEVICE_SHAPE_TENSOR) {
        at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(
            device_shape_tensor_size);
      } else if ((4 == size.size()) && mem_format.has_value()) {
        at_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
            size, CalculateStrides(size, mem_format.value()));
      } else if ((5 == size.size()) && mem_format.has_value()) {
        at_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
            size, CalculateStrides5d(size, mem_format.value()));
      } else {
        at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
      }

      // Keep a pointer to the storageless tensor from the internal tensor
      auto at_internal_impl = GetHbInternalTensorImpl(at_internal_tensor);
      HABANA_ASSERT(at_internal_impl != nullptr);

      // As its an inplace op and we want this op to execute
      // we want to wind back status of this tensor to registered
      // so that when post order is created, we actually execute it
      // auto context =
      //    habana_lazy_executor.getDeviceExecutionContext(
      //        options.device().index());
      // context->MarkTensorStatus(
      //    hb_tensor.getTensorUniqueId(), LazyTensorExecutionStatus::kINPUT);
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
    Tensor at_tensor = AtenFromHbLazyTensor(hb_tensor);
    // Setup the tensor sizes & strides for tensor with dim = 4, else for now
    // assuming contiguous
    if ((4 == size.size()) && mem_format.has_value()) {
      at_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          size, CalculateStrides(size, mem_format.value()));
    } else if ((5 == size.size()) && mem_format.has_value()) {
      at_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          size, CalculateStrides5d(size, mem_format.value()));
    } else {
      at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }
    return at_tensor;
  }
}

Tensor empty_strided_hpu_lazy(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options,
    bool create_storage,
    synTensorType tensor_type,
    int64_t storage_offset) {
  PT_LAZY_TRACE;
  at::Tensor empty_tensor =
      empty_hpu_lazy(size, options, c10::nullopt, create_storage, tensor_type);
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
      hl_empty.value().getAttachedTensorImpl()->set_sizes_and_strides(
          size, stride);
    }
  }
  return empty_tensor;
}

Tensor clone_hpu_lazy(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format) {
  PT_LAZY_TRACE;
  static_cast<void>(memory_format);
  TORCH_CHECK(self.defined(), "src is undefined");
  TORCH_CHECK(
      self.device().type() == c10::DeviceType::HPU,
      "Lazy kernel only supports clone on Habana Device");
  TORCH_CHECK(
      IsHbLazyTensor(self),
      "src is not a Habana Lazy Tensor, currently NOT supported in cloning");

  auto is_5d_tensor = self.dim() == 5;
  if (memory_format == c10::MemoryFormat::ChannelsLast ||
      memory_format == c10::MemoryFormat::ChannelsLast3d) {
    if (is_5d_tensor) {
      int64_t dim_chl_pos[] = {0, 2, 3, 4, 1};
      at::IntArrayRef chl_pos = dim_chl_pos;
      return permute_cl_hpu_lazy(self, chl_pos);

    } else {
      int64_t dim_chl_pos[] = {0, 2, 3, 1};
      at::IntArrayRef chl_pos = dim_chl_pos;
      return permute_cl_hpu_lazy(self, chl_pos);
    }
  } else {
    // We need to add device to device copy kernel here
    // As d2D copies may not mean trigger execution, we just need to add the
    // nodes like memcopy to our lazy graph that we are creating
    HbLazyTensor hb_tensor = GetOrCreateHbLazyTensor(self, self.device());
    hb_tensor = HandleViewsOrUpdate(self, hb_tensor);
    auto node = ir::Node::Create(
        Symbol::fromQualString("hpu::habana_d2d_memcpy"),
        {hb_tensor.GetIrValue()});
    auto result = empty_hpu_lazy(
        self.sizes(),
        self.options(),
        self.suggest_memory_format(),
        /*storage=*/false);
    auto hlresult = GetHbLazyTensor(result);
    auto layout_format = hb_tensor.GetTensorLayout();
    hlresult.SetTensorLayout(layout_format);
    ir::Value& out = hlresult.CurrentIrValue();
    out.SetNode(
        node,
        hlresult.GetDevice(),
        hlresult.GetSizes(),
        hlresult.dtype_optional());
    updateDstDependencies(hlresult, result);
    std::vector<at::Tensor> input_pt_vec{self};
    node->AddInputPtTensors(input_pt_vec);
    flush_op(result);
    return result;
  }
}

Tensor& zero_hpu_lazy(Tensor& self) {
  PT_LAZY_TRACE;
  return fill_hpu_lazy_(self, 0);
}
Tensor cat_hpu_lazy(const TensorList tensors, int64_t dim_) {
  PT_LAZY_TRACE;

  // handle views
  auto t_list = HandleViewsTensorList(tensors);

  const TensorList view_list{t_list};

  struct Kernel : public LazyOp<at::Tensor> {
    explicit Kernel(const at::TensorList tensors, int64_t dim)
        : LazyOp<at::Tensor>("aten::cat", {tensors, dim}, {1}, {}, -1),
          tensors{tensors},
          dim{dim} {}
    at::Tensor get_result_overrideable() override {
      auto first_tensor = tensors[0];

      auto shape_out = CatOutOperator::compute_output_shape(tensors, dim);
      return empty_hpu_lazy(
          shape_out,
          first_tensor.options(),
          first_tensor.suggest_memory_format(),
          false);
    }
    const TensorList tensors;
    int64_t dim;
  };
  Kernel k{view_list, dim_};
  return k.call();
}

Tensor& cat_hpu_lazy_out(
    Tensor& result,
    const TensorList tensors,
    int64_t dim_) {
  PT_LAZY_TRACE;

  auto out_size = CatOutOperator::compute_output_shape(tensors, dim_);
  LazyOp<at::Tensor&> k{"aten::cat", {tensors, dim_, result}, {out_size}};
  return k.call(result);
}

Tensor transpose_hpu_lazy(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  PT_LAZY_TRACE;
  ir::NodePtr node = std::make_shared<ir::Transpose>(self, dim0_, dim1_);
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      TransposeOperator::compute_output_shape(self, dim0_, dim1_);
  auto result =
      empty_strided_hpu_lazy(new_sizes, new_strides, self.options(), false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor& transpose_hpu_lazy_(Tensor& self, int64_t dim0_, int64_t dim1_) {
  PT_LAZY_TRACE;
  return AtenHpuTypeDefault::transpose_(self, dim0_, dim1_);
}

Tensor t_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
  hl_self = HandleViewsOrUpdate(self, hl_self);
  auto node = ir::Node::Create(
      Symbol::fromQualString("aten::t"), {hl_self.GetIrValue()});

  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) = TOperator::compute_output_shape(self);
  auto result =
      empty_strided_hpu_lazy(new_sizes, new_strides, self.options(), false);

  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor& t_hpu_lazy_(Tensor& self) {
  // return t_hpu_(self);
  PT_LAZY_TRACE;
  return AtenHpuTypeDefault::t_(self);
}

void adjustPTSizesLazy(Tensor& t) {
  // PT expects metadata like sizes and strides same as in NCHW,
  // but data permuted for channel last, so change the size and stride
  // NCHW
  auto sizes = t.sizes().vec();
  std::vector<int> out_pos = {0, 3, 1, 2};
  std::vector<long int> swapped_sizes = {
      sizes[out_pos[0]],
      sizes[out_pos[1]],
      sizes[out_pos[2]],
      sizes[out_pos[3]]};
  std::vector<int> out_pos_5d = {0, 4, 1, 2, 3};
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
  // For 4D tensors we need to make sure that we generate the PT channel last
  // strides. Also as its a front end tensor, there may be a backend tensor
  // already if so, change dims for that tensor too.
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
Tensor permute_cl_hpu_lazy(const Tensor& self, IntArrayRef dims_) {
  PT_LAZY_TRACE;
  ir::NodePtr node = std::make_shared<ir::PermuteCL>(self, dims_);
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      PermuteOperator::compute_output_shape(self, dims_.vec());
  auto result =
      empty_strided_hpu_lazy(new_sizes, new_strides, self.options(), false);
  adjustPTSizesLazy(result);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  return result;
}

Tensor permute_hpu_lazy(const Tensor& self, IntArrayRef dims_in) {
  PT_LAZY_TRACE;
  auto dims_vec = dims_in.vec();
  for (unsigned i = 0; i < dims_in.size(); i++) {
    dims_vec[i] =
        at::maybe_wrap_dim(dims_in[i], self.dim(), /*wrap_scalar=*/true);
  }
  IntArrayRef dims_(dims_vec);
  ir::NodePtr node = std::make_shared<ir::Permute>(self, dims_);
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      PermuteOperator::compute_output_shape(self, dims_.vec());
  auto result =
      empty_strided_hpu_lazy(new_sizes, new_strides, self.options(), false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor expand_hpu_lazy(const Tensor& self, IntArrayRef size_in, bool implicit) {
  PT_LAZY_TRACE;
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
  auto size = size_in;
  std::vector<int64_t> initvec{1};
  size = (size_in.vec().size() == 0) ? initvec : size_in;

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) =
      at::inferExpandGeometry(self.sizes(), self.strides(), size);

  // expandedStrides will be set to 0 by inferExpandGeometry.
  // Since we give back a contiguous tensor, we will set strides
  // to proper values.
  habana_helpers::recalc_strides(expandedStrides, expandedSizes);
  ir::NodePtr node;

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    auto expand_shape_tensor = empty_strided_hpu_lazy(
        expandedSizes, expandedStrides, self.options(), false, SHAPE_TENSOR);
    node = std::make_shared<ir::Expand>(self, expand_shape_tensor, implicit);
  } else {
    node = std::make_shared<ir::Expand>(self, size, implicit);
  }

  auto result = empty_strided_hpu_lazy(
      expandedSizes, expandedStrides, self.options(), false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());

  flush_op(result);
  return result;
}

std::vector<Tensor> split_with_sizes_hpu_lazy(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  PT_LAZY_TRACE;
  auto node =
      std::make_shared<habana_lazy::ir::SplitWithSize>(self, split_sizes, dim);
  auto shapes =
      SplitWithSizeOperator::compute_output_shape(self, split_sizes, dim);

  int64_t i = 0;
  std::vector<at::Tensor> result(shapes.size());

  // This ZST output tensor should ideally be handled at Synapse level, but
  // since it is throwing errors in that case we are forced to add this
  // work-around.
  // TBD: Investigate and raise a JIRA on GC.
  auto sum_elm = habana_helpers::tensor_numel(self);
  if (!sum_elm) {
    for (const auto& shape : shapes) {
      result[i++] = empty_hpu_lazy(
          shape, self.options(), self.suggest_memory_format(), true);
    }
    std::vector<habana_lazy::HbLazyTensor> hlresult;
    hlresult.reserve(result.size());

    for (const auto& pt : result) {
      hlresult.push_back(habana_lazy::GetHbLazyTensor(pt));
    }
    size_t m_index = 0;
    for (auto ht : hlresult) {
      updateDstDependencies(hlresult[m_index], result[m_index]);
    }
    flush_op(result);
    return result;
  }

  for (const auto& shape : shapes) {
    result[i++] = empty_hpu_lazy(
        shape, self.options(), self.suggest_memory_format(), false);
  }

  std::vector<habana_lazy::HbLazyTensor> hlresult;
  hlresult.reserve(result.size());

  for (const auto& pt : result) {
    hlresult.push_back(habana_lazy::GetHbLazyTensor(pt));
  }

  habana_lazy::ir::Value& out = hlresult[0].CurrentIrValue();
  node->set_as_output_tensor_list();
  out.SetNode(
      node,
      hlresult[0].GetDevice(),
      hlresult[0].GetSizes(),
      hlresult[0].dtype_optional());

  habana_lazy::ir::NodePtr node_unpack =
      std::make_shared<habana_lazy::ir::ListUnpack>(out);

  size_t m_index = 0;

  for (auto ht : hlresult) {
    updateDstDependencies(hlresult[m_index], result[m_index]);
    auto& out = ht.CurrentIrValue();
    out.SetNode(
        node_unpack,
        ht.GetDevice(),
        ht.GetSizes(),
        ht.dtype_optional(),
        m_index++);
  }

  flush_op(result);
  return result;
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
  ir::NodePtr node =
      std::make_shared<ir::TopK>(self, size_dim, dim, descending, true);
  auto shape_out = self.sizes().vec();

  // out 0
  auto result_0 = empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);

  auto hlresult_0 = GetHbLazyTensor(result_0);
  ir::Value& out_0 = hlresult_0.CurrentIrValue();
  out_0.SetNode(
      node,
      hlresult_0.GetDevice(),
      hlresult_0.GetSizes(),
      hlresult_0.dtype_optional());

  // out 1
  auto result_1 = empty_hpu_lazy(
      shape_out,
      self.options().dtype(kInt),
      self.suggest_memory_format(),
      false);
  auto hlresult_1 = GetHbLazyTensor(result_1);
  ir::Value& out_1 = hlresult_1.CurrentIrValue();
  out_1.SetNode(
      node,
      hlresult_1.GetDevice(),
      hlresult_1.GetSizes(),
      hlresult_1.dtype_optional(),
      1);

  flush_op({result_0, result_1});
  return std::make_tuple(result_0, result_1);
}

at::Tensor elu_hpu_lazy(
    const at::Tensor& self,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::elu", {self, alpha, scale, input_scale}};
  return k.call();
}

at::Tensor& elu_hpu_lazy_(
    at::Tensor& self,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::elu_", {self, alpha, scale, input_scale}};
  return k.call(self);
}

at::Tensor& leaky_relu_lazy_(
    at::Tensor& self,
    const at::Scalar& negative_slope) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::leaky_relu_", {self, negative_slope}};
  return k.call(self);
}

at::Tensor leaky_relu_backward_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negative_slope,
    bool self_is_result) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{
      "aten::leaky_relu_backward",
      {grad_output, self, negative_slope, self_is_result},
      {2, 3}};
  return k.call();
}

at::Tensor leaky_relu_lazy(
    const at::Tensor& self,
    const at::Scalar& negative_slope) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::leaky_relu", {self, negative_slope}};
  return k.call();
}

at::Tensor flip_hpu_lazy(const at::Tensor& self, at::IntArrayRef dims) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k("aten::flip", {self, dims});
  return k.call();
}

at::Tensor diag_hpu_lazy(const at::Tensor& self, int64_t diagonal) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k(
      "aten::diag",
      {self, diagonal},
      {},
      {DiagOperator::compute_output_shape(self, diagonal)});
  return k.call();
}

at::Tensor& diag_hpu_lazy_out(
    const at::Tensor& self,
    int64_t diagonal,
    at::Tensor& out) {
  PT_LAZY_TRACE;
  LazyOp<Tensor&> k(
      "hpu::diag_out",
      {self, diagonal, out},
      {},
      {DiagOutOperator::compute_output_shape(self, diagonal)});
  return k.call(out);
}

Tensor sign_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::sign", {input}};
  return k.call();
}

Tensor& sign_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::sign", {input}};
  return k.call(input);
}

Tensor sgn_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  TORCH_CHECK(!input.is_complex(), "Unsupported complex data type provided");
  return sign_hpu_lazy(input);
}

Tensor& sgn_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  TORCH_CHECK(!input.is_complex(), "Unsupported complex data type provided");
  return sign_hpu_lazy_(input);
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

  shape.push_back(num_classes);
  Tensor ret = at::zeros(shape, self.options());
  ret.scatter_(-1, self.unsqueeze(-1), 1);
  return ret;
}

Tensor upsample_nearest2d_hpu_lazy(
    const Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
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
  ir::NodePtr node = std::make_shared<ir::UpsampleNearest2d>(
      input_cast, output_size, scale_factors);

  auto memory_format = input_cast.suggest_memory_format();
  auto shape_out = UpsampleOperator::compute_output_shape(
      input_cast.sizes().vec(), output_size, scale_factors, memory_format);
  auto result =
      empty_hpu_lazy(shape_out, input_cast.options(), memory_format, false);
  auto hlresult = GetHbLazyTensor(result);
  ir::Value& out = hlresult.CurrentIrValue();
  out.SetNode(
      node,
      hlresult.GetDevice(),
      hlresult.GetSizes(),
      hlresult.dtype_optional());
  // updatet the view if any
  updateDstDependencies(hlresult, result);
  flush_op(result);
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
    c10::optional<at::IntArrayRef> output_size,
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
  auto memory_format = grad_output_cast.suggest_memory_format();
  std::vector<int64_t> permuted_sizes = input_size.vec();
  permuted_sizes[0] = input_size[0];
  permuted_sizes[1] = input_size[2];
  permuted_sizes[2] = input_size[3];
  permuted_sizes[3] = input_size[1];

  ir::NodePtr node = std::make_shared<ir::UpsampleNearest2dBackward>(
      grad_output_cast, output_size, permuted_sizes, scale_factors);

  auto result = empty_hpu_lazy(
      input_size, grad_output_cast.options(), memory_format, false);
  auto hlresult = GetHbLazyTensor(result);
  ir::Value& out = hlresult.CurrentIrValue();
  out.SetNode(
      node,
      hlresult.GetDevice(),
      hlresult.GetSizes(),
      hlresult.dtype_optional());
  // updatet the view if any
  updateDstDependencies(hlresult, result);
  flush_op(result);
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
    c10::optional<at::IntArrayRef> output_size,
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
    c10::optional<at::IntArrayRef> output_size,
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
  permuted_sizes[0] = input_size[0];
  permuted_sizes[1] = input_size[2];
  permuted_sizes[2] = input_size[3];
  permuted_sizes[3] = input_size[4];
  permuted_sizes[4] = input_size[1];
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

at::Tensor& hardsigmoid_hpu_lazy_(at::Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::hardsigmoid_", {self}};
  return k.call(self);
}

Tensor hardsigmoid_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::hardsigmoid", {input}};
  return k.call();
}

Tensor hardsigmoid_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::hardsigmoid_backward", {grad_output, self}};
  return k.call();
}

Tensor& tanh_out_hpu_lazy(Tensor& out, const Tensor& self) {
  PT_LAZY_TRACE;
  HABANA_ASSERT(0);
  return tanh_out_hpu(out, self);
}

Tensor gelu_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  auto hl_input = GetOrCreateHbLazyTensor(self, c10::kHPU);
  auto node = ir::Node::Create(
      Symbol::fromQualString("aten::gelu"), {hl_input.GetIrValue()});
  auto result = empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);
  flush_op(result);
  return result;
}

Tensor gelu_backward_hpu_lazy(const Tensor& grad, const Tensor& self) {
  PT_LAZY_TRACE;
  auto hl_grad = GetOrCreateHbLazyTensor(grad, c10::kHPU);
  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);

  auto node = ir::Node::Create(
      Symbol::fromQualString("aten::gelu_backward"),
      {hl_grad.GetIrValue(), hl_self.GetIrValue()});

  auto result = empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{grad, self};
  node->AddInputPtTensors(input_pt_vec);
  flush_op(result);
  return result;
}

Tensor& exp_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::exp_", {self}};
  return k.call(self);
}

Tensor exp_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::exp", {self}};
  return k.call();
}

Tensor& neg_out_hpu_lazy(Tensor& result, const Tensor& input) {
  PT_LAZY_TRACE;
  HABANA_ASSERT(0);
  return neg_out_hpu(result, input);
}

Tensor& reciprocal_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::reciprocal_", {self}};
  return k.call(self);
}

Tensor reciprocal_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::reciprocal", {self}};
  return k.call();
}

Tensor& reciprocal_out_hpu_lazy(Tensor& result, const Tensor& self) {
  PT_LAZY_TRACE;
  HABANA_ASSERT(0);
  return reciprocal_out_hpu(result, self);
}

Tensor isfinite_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k_{
      "aten::isfinite", {input}, {input.sizes().vec()}, c10::ScalarType::Bool};
  return k_.call();
}

Tensor abs_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::abs", {input}};
  return k.call();
}

Tensor& abs_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::abs_", {self}};
  return k.call(self);
}

Tensor neg_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::neg", {self}};
  return k.call();
}
Scalar _local_scalar_dense_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  Scalar out;
  // If self is a lazy tensor make sure the execution till the point of self
  // getting flled has finished before we start copying
  if (IsHbLazyTensor(self)) {
    HbLazyTensor hb_tensor = GetOrCreateHbLazyTensor(self, self.device());
    hb_tensor = HandleViewsOrUpdate(self, hb_tensor);
    if (self.device().type() == c10::DeviceType::HPU) {
      // Trigger point execution
      HbLazyTensor::StepMarker({});
    }
    // if there is a view, we need to sync before accessing the tensor_data.
    // This is because we skip view outputs in stepmarker
    hb_tensor = GetHbLazyTensor(HandleViewsD2H(self));
    auto tensor_data = hb_tensor.GetHbLazyTensorData();
    out = habana_helpers::_local_scalar_dense_internal(tensor_data.value());
  } else {
    out = habana_helpers::_local_scalar_dense_internal(self);
  }
  return out;
}
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_lazy(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor,
    float mom,
    bool nesterov) {
  PT_LAZY_TRACE;
  ir::NodePtr node = std::make_shared<ir::OptimizerSparseSgdValidCount>(
      gradients,
      weights_in,
      moments_in,
      indices,
      learning_rate,
      valid_count_tensor,
      mom,
      nesterov);

  auto hlweights = GetHbLazyTensor(weights_in);
  ir::Value& out1 = hlweights.CurrentIrValue();
  out1.SetNode(
      node,
      hlweights.GetDevice(),
      hlweights.GetSizes(),
      hlweights.dtype_optional());
  auto hlmoments = GetHbLazyTensor(moments_in);
  ir::Value& out2 = hlmoments.CurrentIrValue();
  out2.SetNode(
      node,
      hlmoments.GetDevice(),
      hlmoments.GetSizes(),
      hlmoments.dtype_optional(),
      1);
  flush_op({weights_in, moments_in});
  return std::tie(weights_in, moments_in);
}
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_lazy(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor) {
  PT_LAZY_TRACE;
  auto node = ir::Node::Create(
      Symbol::fromQualString("hpu::habanaOptimizerSparseAdagrad"), {});

  std::vector<HbLazyTensor> hl_tensors;
  hl_tensors.push_back(GetOrCreateHbLazyTensor(gradients, c10::kHPU));
  hl_tensors.push_back(GetOrCreateHbLazyTensor(weights_in, c10::kHPU));
  hl_tensors.push_back(GetOrCreateHbLazyTensor(moments_in, c10::kHPU));
  hl_tensors.push_back(GetOrCreateHbLazyTensor(indices, c10::kHPU));
  hl_tensors.push_back(GetOrCreateHbLazyTensor(learning_rate, c10::kHPU));
  hl_tensors.push_back(GetOrCreateHbLazyTensor(valid_count_tensor, c10::kHPU));

  for (auto& i : hl_tensors) {
    node->AddInput(i.GetIrValue());
  }

  auto hlweights = GetHbLazyTensor(weights_in);
  ir::Value& out1 = hlweights.CurrentIrValue();
  out1.SetNode(
      node,
      hlweights.GetDevice(),
      hlweights.GetSizes(),
      hlweights.dtype_optional());
  auto hlmoments = GetHbLazyTensor(moments_in);
  ir::Value& out2 = hlmoments.CurrentIrValue();
  out2.SetNode(
      node,
      hlmoments.GetDevice(),
      hlmoments.GetSizes(),
      hlmoments.dtype_optional(),
      1);

  std::vector<at::Tensor> input_pt_vec{
      gradients,
      weights_in,
      moments_in,
      indices,
      learning_rate,
      valid_count_tensor};
  node->AddInputPtTensors(input_pt_vec);

  flush_op({weights_in, moments_in});
  return std::tie(weights_in, moments_in);
}

void optimizer_adamw_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& exp_avg,
    TensorList& exp_avg_sq,
    at::Tensor& lr_t,
    at::Tensor& neg_step_t,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float modified_wd) {
  PT_LAZY_TRACE;
  std::vector<at::Tensor> pweights;
  for (size_t i = 0; i < weights.size(); i++) {
    auto weight_hwck = permute_wt_hpu(weights[i]);
    pweights.push_back(weight_hwck);
  }
  TensorList weights_hwck(pweights);

  for (size_t i = 0; i < weights_hwck.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights_hwck[i]);
    updateDstDependencies(hlweight, weights_hwck[i], true);
  }

  auto hl_lr_t = GetOrCreateHbLazyTensor(lr_t, c10::kHPU);
  auto hl_neg_step_t = GetOrCreateHbLazyTensor(neg_step_t, c10::kHPU);

  ir::NodePtr node = std::make_shared<ir::OptimizerFusedAdamw>(
      gradients,
      weights_hwck,
      exp_avg,
      exp_avg_sq,
      lr_t,
      neg_step_t,
      beta1,
      beta2,
      epsilon,
      modified_wd);

  int64_t out_index = 0;

  auto hlweight = habana_lazy::GetHbLazyTensor(weights_hwck[0]);
  habana_lazy::ir::Value& out = hlweight.CurrentIrValue();
  node->set_as_output_tensor_list();
  out.SetNode(
      node,
      hlweight.GetDevice(),
      hlweight.GetSizes(),
      hlweight.dtype_optional());

  habana_lazy::ir::NodePtr node_unpack =
      std::make_shared<habana_lazy::ir::ListUnpack>(out);

  for (size_t i = 0; i < weights_hwck.size(); i++) {
    if (modified_wd != 1.0) {
      auto hl_wd = GetHbLazyTensor(weights_hwck[i]);
      ir::Value& out0 = hl_wd.CurrentIrValue();
      out0.SetNode(
          node_unpack,
          hl_wd.GetDevice(),
          hl_wd.GetSizes(),
          hl_wd.dtype_optional(),
          out_index++);
    }

    auto hl_exp_avg = GetHbLazyTensor(exp_avg[i]);
    ir::Value& out1 = hl_exp_avg.CurrentIrValue();
    out1.SetNode(
        node_unpack,
        hl_exp_avg.GetDevice(),
        hl_exp_avg.GetSizes(),
        hl_exp_avg.dtype_optional(),
        out_index++);

    auto hl_exp_avg_1 = GetHbLazyTensor(exp_avg[i]);
    ir::Value& out2 = hl_exp_avg_1.CurrentIrValue();
    out2.SetNode(
        node_unpack,
        hl_exp_avg_1.GetDevice(),
        hl_exp_avg_1.GetSizes(),
        hl_exp_avg_1.dtype_optional(),
        out_index++);

    auto hl_exp_avg_sq = GetHbLazyTensor(exp_avg_sq[i]);
    ir::Value& out3 = hl_exp_avg_sq.CurrentIrValue();
    out3.SetNode(
        node_unpack,
        hl_exp_avg_sq.GetDevice(),
        hl_exp_avg_sq.GetSizes(),
        hl_exp_avg_sq.dtype_optional(),
        out_index++);

    auto hl_exp_avg_sq_1 = GetHbLazyTensor(exp_avg_sq[i]);
    ir::Value& out4 = hl_exp_avg_sq_1.CurrentIrValue();
    out4.SetNode(
        node_unpack,
        hl_exp_avg_sq_1.GetDevice(),
        hl_exp_avg_sq_1.GetSizes(),
        hl_exp_avg_sq_1.dtype_optional(),
        out_index++);

    auto hl_weight = GetHbLazyTensor(weights_hwck[i]);
    ir::Value& out5 = hl_weight.CurrentIrValue();
    out5.SetNode(
        node_unpack,
        hl_weight.GetDevice(),
        hl_weight.GetSizes(),
        hl_weight.dtype_optional(),
        out_index++);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    HbLazyTensor::StepMarker({});
  }
}

Tensor fused_norm_hpu_lazy(
    std::vector<Tensor>& grad,
    const Tensor& max_norm,
    float norm_type) {
  PT_LAZY_TRACE;
  for (size_t i = 0; i < grad.size(); i++) {
    auto hlweight = GetHbLazyTensor(grad[i]);
    updateDstDependencies(hlweight, grad[i], true);
  }
  ir::NodePtr node = std::make_shared<ir::FusedNorm>(grad, max_norm, norm_type);
  int64_t out_index = 0;

  auto hlgrad = habana_lazy::GetHbLazyTensor(grad[0]);
  habana_lazy::ir::Value& out1 = hlgrad.CurrentIrValue();
  node->set_as_output_tensor_list();
  out1.SetNode(
      node, hlgrad.GetDevice(), hlgrad.GetSizes(), hlgrad.dtype_optional());

  habana_lazy::ir::NodePtr node_unpack =
      std::make_shared<habana_lazy::ir::ListUnpack>(out1);

  auto result = empty_hpu_lazy(
      {1}, grad[0].options(), grad[0].suggest_memory_format(), false);

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
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  for (size_t i = 0; i < grad.size(); i++) {
    auto grad_t = grad[i];
    auto hlgrad = GetHbLazyTensor(grad_t);
    auto id = hlgrad.getTensorUniqueId();
    auto it = context->view_table.find(id);

    if (it == context->view_table.end()) {
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

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    HbLazyTensor::StepMarker({});
  }

  return result;
}

Tensor optimizer_lamb_fused_norm_hpu_lazy(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm) {
  PT_LAZY_TRACE;
  auto clip_norm = get_tensor_for_scalar(1.0);
  ir::NodePtr node =
      std::make_shared<ir::LambFusedNorm>(grad, max_grad_norm, clip_norm);

  auto result = empty_hpu_lazy(
      {1}, grad[0].options(), grad[0].suggest_memory_format(), false);
  auto hlresult = GetHbLazyTensor(result);
  ir::Value& out = hlresult.CurrentIrValue();
  out.SetNode(
      node,
      hlresult.GetDevice(),
      hlresult.GetSizes(),
      hlresult.dtype_optional());

  updateDstDependencies(hlresult, result);

  return result;
}

std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
optimizer_lamb_phase1_hpu_lazy(
    const std::vector<at::Tensor>& gradients,
    std::vector<at::Tensor>& weights,
    std::vector<at::Tensor>& exp_avg,
    std::vector<at::Tensor>& exp_avg_sq,
    const at::Tensor& clip_global_grad_norm,
    const int grad_averaging,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay) {
  PT_LAZY_TRACE;
  static_cast<void>(lr);

  /*
  Assuming mark step is present before optimizer is invoked
  for (size_t i = 0; i < weights.size(); i++) {
    auto hl_grad = GetHbLazyTensor(gradients[i]);
    updateDstDependencies(hl_grad, gradients[i], true);

    auto hl_wts = GetHbLazyTensor(weights[i]);
    updateDstDependencies(hl_wts, weights[i], true);

    auto hlexpavg = GetHbLazyTensor(exp_avg[i]);
    updateDstDependencies(hlexpavg, exp_avg[i], true);

    auto hlexpavgsq = GetHbLazyTensor(exp_avg_sq[i]);
    updateDstDependencies(hlexpavgsq, exp_avg_sq[i], true);
  }
  */

  auto hl_clip_global = GetHbLazyTensor(clip_global_grad_norm);
  updateDstDependencies(hl_clip_global, clip_global_grad_norm, true);

  float bias_correction1 = 1.0, bias_correction2 = 1.0;
  if (bias_correction) {
    bias_correction1 = 1.0 - std::pow(beta1, step);
    bias_correction2 = 1.0 - std::pow(beta2, step);
  }

  float beta3 = 1.0;
  if (grad_averaging) {
    beta3 = 1 - beta1;
  }

  auto bias_correction1_t = get_tensor_for_scalar(bias_correction1);
  auto bias_correction2_t = get_tensor_for_scalar(bias_correction2);

  ir::NodePtr node = std::make_shared<ir::OptimizerFusedLambPhase1>(
      gradients,
      weights,
      exp_avg,
      exp_avg_sq,
      clip_global_grad_norm,
      beta1,
      beta2,
      beta3,
      epsilon,
      bias_correction1_t,
      bias_correction2_t,
      weight_decay);

  int64_t out_index = 0;

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);

  std::vector<Tensor> weight_norm_vec, adam_norm_vec, adam_step_vec;
  for (size_t i = 0; i < weights.size(); i++) {
    auto adam_step = empty_hpu_lazy(
        weights[i].sizes(),
        weights[i].options(),
        weights[i].suggest_memory_format(),
        false);
    auto hl_adam_step = GetHbLazyTensor(adam_step);
    ir::Value& out1 = hl_adam_step.CurrentIrValue();
    out1.SetNode(
        node,
        hl_adam_step.GetDevice(),
        hl_adam_step.GetSizes(),
        hl_adam_step.dtype_optional(),
        out_index++);

    context->m_retained_tensor_list.emplace_back(adam_step);
    adam_step_vec.push_back(adam_step);

    auto adam_norm = empty_hpu_lazy(
        {1}, weights[i].options(), weights[i].suggest_memory_format(), false);
    auto hl_adam_norm = GetHbLazyTensor(adam_norm);
    ir::Value& out2 = hl_adam_norm.CurrentIrValue();
    out2.SetNode(
        node,
        hl_adam_norm.GetDevice(),
        hl_adam_norm.GetSizes(),
        hl_adam_norm.dtype_optional(),
        out_index++);

    context->m_retained_tensor_list.emplace_back(adam_norm);
    adam_norm_vec.push_back(adam_norm);

    auto weight_norm = empty_hpu_lazy(
        {1}, weights[i].options(), weights[i].suggest_memory_format(), false);
    auto hl_weight_norm = GetHbLazyTensor(weight_norm);
    ir::Value& out3 = hl_weight_norm.CurrentIrValue();
    out3.SetNode(
        node,
        hl_weight_norm.GetDevice(),
        hl_weight_norm.GetSizes(),
        hl_weight_norm.dtype_optional(),
        out_index++);

    context->m_retained_tensor_list.emplace_back(weight_norm);
    weight_norm_vec.push_back(weight_norm);

    // add the tensors that are updated inplace
    auto exp_avg_temp = empty_hpu_lazy(
        exp_avg[i].sizes(),
        exp_avg[i].options(),
        exp_avg[i].suggest_memory_format(),
        false);
    auto hl_exp_avg_temp = GetHbLazyTensor(exp_avg_temp);
    ir::Value& out4 = hl_exp_avg_temp.CurrentIrValue();
    out4.SetNode(
        node,
        hl_exp_avg_temp.GetDevice(),
        hl_exp_avg_temp.GetSizes(),
        hl_exp_avg_temp.dtype_optional(),
        out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg_temp);

    auto hl_exp_avg = GetHbLazyTensor(exp_avg[i]);
    ir::Value& out5 = hl_exp_avg.CurrentIrValue();
    out5.SetNode(
        node,
        hl_exp_avg.GetDevice(),
        hl_exp_avg.GetSizes(),
        hl_exp_avg.dtype_optional(),
        out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg[i]);

    auto exp_avg_sq_temp = empty_hpu_lazy(
        exp_avg_sq[i].sizes(),
        exp_avg_sq[i].options(),
        exp_avg_sq[i].suggest_memory_format(),
        false);
    auto hl_exp_avg_sq_temp = GetHbLazyTensor(exp_avg_sq_temp);
    ir::Value& out6 = hl_exp_avg_sq_temp.CurrentIrValue();
    out6.SetNode(
        node,
        hl_exp_avg_sq_temp.GetDevice(),
        hl_exp_avg_sq_temp.GetSizes(),
        hl_exp_avg_sq_temp.dtype_optional(),
        out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg_sq_temp);

    auto hl_exp_avg_sq = GetHbLazyTensor(exp_avg_sq[i]);
    ir::Value& out7 = hl_exp_avg_sq.CurrentIrValue();
    out7.SetNode(
        node,
        hl_exp_avg_sq.GetDevice(),
        hl_exp_avg_sq.GetSizes(),
        hl_exp_avg_sq.dtype_optional(),
        out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg_sq[i]);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    HbLazyTensor::StepMarker({});
  }

  return std::tie(weight_norm_vec, adam_norm_vec, adam_step_vec);
}

void optimizer_lamb_phase2_hpu_lazy(
    std::vector<at::Tensor>& weights,
    const std::vector<at::Tensor>& adam_norm,
    const std::vector<at::Tensor>& weight_norm,
    const std::vector<at::Tensor>& adam_step,
    const std::vector<at::Tensor>& trust_ratio,
    const float step,
    const float weight_decay,
    const int use_lamb) {
  PT_LAZY_TRACE;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hl_weights = GetHbLazyTensor(weights[i]);
    updateDstDependencies(hl_weights, weights[i], true);
  }

  auto nstep_t = at::tensor(-step).to(c10::kHPU, true);

  // Build Params for the graph
  ir::NodePtr node = std::make_shared<ir::OptimizerFusedLambPhase2>(
      weights,
      adam_norm,
      weight_norm,
      adam_step,
      trust_ratio,
      nstep_t,
      weight_decay,
      use_lamb);

  int64_t out_index = 0;
  for (size_t i = 0; i < weights.size(); i++) {
    auto hl_weight = GetHbLazyTensor(weights[i]);
    ir::Value& out1 = hl_weight.CurrentIrValue();
    out1.SetNode(
        node,
        hl_weight.GetDevice(),
        hl_weight.GetSizes(),
        hl_weight.dtype_optional(),
        out_index++);
  }
}

Tensor& optimizer_adagrad_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  PT_LAZY_TRACE;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights[i]);
    updateDstDependencies(hlweight, weights[i], true);

    auto hlvariance = GetHbLazyTensor(variances[i]);
    updateDstDependencies(hlvariance, variances[i], true);
  }

  ir::NodePtr node = std::make_shared<ir::OptimizerFusedAdagrad>(
      gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);

  int64_t out_index = 0;
  HABANA_ASSERT(weights.size() == variances.size());

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights[i]);
    ir::Value& out1 = hlweight.CurrentIrValue();
    out1.SetNode(
        node,
        hlweight.GetDevice(),
        hlweight.GetSizes(),
        hlweight.dtype_optional(),
        out_index++);

    auto hlvariance = GetHbLazyTensor(variances[i]);
    ir::Value& out2 = hlvariance.CurrentIrValue();
    out2.SetNode(
        node,
        hlvariance.GetDevice(),
        hlvariance.GetSizes(),
        hlvariance.dtype_optional(),
        out_index++);
  }

  flush_op(lr);
  return lr;
}

Tensor& optimizer_sgd_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_LAZY_TRACE;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights[i]);
    updateDstDependencies(hlweight, weights[i], true);
  }

  ir::NodePtr node = std::make_shared<ir::OptimizerFusedSGD>(
      gradients, weights, lr, wd, mom, damp, nesterov);

  int64_t out_index = 0;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights[i]);
    ir::Value& out1 = hlweight.CurrentIrValue();
    out1.SetNode(
        node,
        hlweight.GetDevice(),
        hlweight.GetSizes(),
        hlweight.dtype_optional(),
        out_index++);
  }

  flush_op(lr);
  return lr;
}

Tensor& optimizer_sgd_momentum_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_LAZY_TRACE;
  std::vector<at::Tensor> pweights;
  for (size_t i = 0; i < weights.size(); i++) {
    auto weight_hwck = permute_wt_hpu(weights[i]);
    pweights.push_back(weight_hwck);
  }
  TensorList weights_hwck(pweights);

  for (size_t i = 0; i < weights_hwck.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights_hwck[i]);
    updateDstDependencies(hlweight, weights_hwck[i], true);

    auto hlmomentum = GetHbLazyTensor(momentum[i]);
    updateDstDependencies(hlmomentum, momentum[i], true);
  }

  ir::NodePtr node = std::make_shared<ir::OptimizerFusedSGDMomentum>(
      gradients,
      weights_hwck,
      momentum,
      epoch_num,
      lr,
      wd,
      mom,
      damp,
      nesterov);

  int64_t out_index = 0;
  HABANA_ASSERT(weights_hwck.size() == momentum.size());

  auto hlweight = GetHbLazyTensor(weights_hwck[0]);
  ir::Value& out = hlweight.CurrentIrValue();
  node->set_as_output_tensor_list();
  out.SetNode(
      node,
      hlweight.GetDevice(),
      hlweight.GetSizes(),
      hlweight.dtype_optional());

  ir::NodePtr node_unpack = std::make_shared<ir::ListUnpack>(out);

  for (size_t i = 0; i < weights_hwck.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights_hwck[i]);
    ir::Value& out1 = hlweight.CurrentIrValue();
    out1.SetNode(
        node_unpack,
        hlweight.GetDevice(),
        hlweight.GetSizes(),
        hlweight.dtype_optional(),
        out_index++);

    auto hlmomentum = GetHbLazyTensor(momentum[i]);
    ir::Value& out2 = hlmomentum.CurrentIrValue();
    out2.SetNode(
        node_unpack,
        hlmomentum.GetDevice(),
        hlmomentum.GetSizes(),
        hlmomentum.dtype_optional(),
        out_index++);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    HbLazyTensor::StepMarker({});
  }
  return lr;
}

at::Tensor ones_like_hpu_lazy(
    const Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
  // Note that currently we are not lowering parameters as per the ones_like
  // schema. This works for the ones_like usage in MNIST (where it is used
  // only for filling grad_out tensor with 1's), but we may need to revisit
  // this in future.
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op(
      "aten::ones_like",
      {self, dtype, layout, device, pin_memory, memory_format},
      {1, 2, 3, 4, 5});
  return op.call();
}

Tensor& bitwise_and_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;
  auto hl_out = GetOrCreateHbLazyTensor(out, c10::kHPU);
  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
  auto hl_other = GetOrCreateHbLazyTensor(other, c10::kHPU);
  auto out_shape = BitwiseOutOperator::compute_output_shape(self, other);
  // Resize output tensor(s) to correct shape if required
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = hl_out.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));
  }
  auto node = ir::Node::Create(
      Symbol::fromQualString("hpu::bitwise_and_Tensor_out"),
      {hl_out.GetIrValue(), hl_self.GetIrValue(), hl_other.GetIrValue()});
  ir::Value& output = hl_out.CurrentIrValue();
  output.SetNode(
      node, hl_out.GetDevice(), hl_out.GetSizes(), hl_out.dtype_optional());
  // updatet the view if any
  updateDstDependencies(hl_out, out);
  std::vector<at::Tensor> input_pt_vec{out, self, other};
  node->AddInputPtTensors(input_pt_vec);
  flush_op(out);
  return out;
}

Tensor& bitwise_and_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    Scalar other) {
  PT_LAZY_TRACE;
  HABANA_ASSERT(false && "Not implemented yet");
  static_cast<void>(out);
  static_cast<void>(self);
  static_cast<void>(other);

  return out;
}

Tensor& bitwise_or_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;
  auto hl_out = GetOrCreateHbLazyTensor(out, c10::kHPU);
  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
  auto hl_other = GetOrCreateHbLazyTensor(other, c10::kHPU);
  auto out_shape = BitwiseOutOperator::compute_output_shape(self, other);
  // Resize output tensor(s) to correct shape if required
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = hl_out.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));
  }
  auto node = ir::Node::Create(
      Symbol::fromQualString("hpu::bitwise_or_Tensor_out"),
      {hl_out.GetIrValue(), hl_self.GetIrValue(), hl_other.GetIrValue()});
  auto hl_result = GetHbLazyTensor(out);
  ir::Value& output = hl_result.CurrentIrValue();
  output.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  // updatet the view if any
  updateDstDependencies(hl_result, out);
  std::vector<at::Tensor> input_pt_vec{out, self, other};
  node->AddInputPtTensors(input_pt_vec);
  return out;
};

Tensor& bitwise_xor_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;

  auto hl_out = GetOrCreateHbLazyTensor(out, c10::kHPU);
  auto out_shape = BitwiseOutOperator::compute_output_shape(self, other);
  // Resize output tensor(s) to correct shape if required
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = hl_out.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));
  }

  LazyOp<at::Tensor&> k{
      "hpu::bitwise_xor_Tensor_out", {out, self, other}, {}, {out_shape}};
  return k.call(out);
};

Tensor& bitwise_not_out_hpu_lazy(Tensor& out, const Tensor& self) {
  PT_LAZY_TRACE;

  auto hl_out = GetOrCreateHbLazyTensor(out, c10::kHPU);
  auto out_shape = self.sizes().vec();
  // Resize output tensor(s) to correct shape if required
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = hl_out.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));
  }

  LazyOp<at::Tensor&> k{
      "hpu::bitwise_not_Tensor_out", {out, self}, {}, {out_shape}};
  return k.call(out);
};

std::tuple<Tensor, Tensor, Tensor> unique2_hpu_lazy(
    const Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  PT_LAZY_TRACE;

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
  auto hl_feature_map = GetHbLazyTensor(feature_map);
  auto hl_valid = GetHbLazyTensor(valid_count);

  // Force an execution here because "unique" is a non shape inferable op.
  std::vector<HbLazyTensor> hl_flush = {hl_feature_map, hl_valid};
  HbLazyTensor::SyncTensorsGraph(&hl_flush);
  auto end = valid_count.item<int64_t>();

  // Add a slice node to capture relevent elements from feature_map
  auto sliced_shape = DimVector{end};
  auto node_slice = std::make_shared<ir::Slice>(feature_map, 0, 0, end, 1);
  auto result = empty_hpu_lazy(
      sliced_shape,
      self.options().dtype(self.scalar_type()),
      self.suggest_memory_format(),
      false);
  auto hl_result = GetOrCreateHbLazyTensor(result, c10::kHPU);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node_slice,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());

  // These are optional tensors which shall be populated only when we
  // start supporting return_inverse and return_counts
  Tensor inverse_indices;
  Tensor counts;

  updateDstDependencies(hl_result, result);
  flush_op(result);
  return std::make_tuple(result, inverse_indices, counts);
};

std::tuple<at::Tensor, at::Tensor> max_dim_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  PT_LAZY_TRACE;
  ir::NodePtr node = std::make_shared<ir::MaxDim>(self, dim, keepdim);

  // Infer Output shape
  auto shape_out = MaxDimOperator::compute_output_shape(self, dim, keepdim);

  auto result = empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto index = empty_hpu_lazy(
      shape_out,
      self.options().dtype(c10::ScalarType::Int),
      self.suggest_memory_format(),
      false);
  auto hl_result1 = GetHbLazyTensor(result);
  auto hl_result2 = GetHbLazyTensor(index);
  ir::Value& out1 = hl_result1.CurrentIrValue();
  ir::Value& out2 = hl_result2.CurrentIrValue();
  out1.SetNode(
      node,
      hl_result1.GetDevice(),
      hl_result1.GetSizes(),
      hl_result1.dtype_optional());
  out2.SetNode(
      node,
      hl_result2.GetDevice(),
      hl_result2.GetSizes(),
      hl_result2.dtype_optional(),
      1);
  updateDstDependencies(hl_result1, result);
  updateDstDependencies(hl_result2, index);
  flush_op({result, index});
  return std::make_tuple(result, index);
}

at::Tensor max_hpu_lazy(const at::Tensor& self) {
  PT_LAZY_TRACE;
  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
  hl_self = HandleViewsOrUpdate(self, hl_self);

  auto node = ir::Node::Create(
      Symbol::fromQualString("aten::max"), {hl_self.GetIrValue()});

  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  // Infer Output shape
  auto shape_out = MaxOperator::compute_output_shape();

  auto result = empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

at::Tensor min_hpu_lazy(const at::Tensor& self) {
  PT_LAZY_TRACE;
  // TODO: Need to make 0-dim output.
  // With ... shape_out = {0}; following error reported
  // Assertion (dst.nbytes() >= src.nbytes()) is false!
  // ... /pytorch-integration/habana_kernels/basic_kernels.cpp:147 copy_hpu_
  // terminate called without an active exception
  // Aborted (core dumped)
  LazyOp<at::Tensor> k{"aten::min", {self}, {}, {{1}}};
  return k.call();
}

Tensor masked_scale_hpu_lazy(
    const Tensor& self,
    const Tensor& mask,
    double scale) {
  PT_LAZY_TRACE;
  // scale changed to support dropout backward based on what we pass for dropout
  scale = scale / (scale - 1);
  auto masked = mul_tensor_hpu_lazy(self, mask);
  auto scaled = mul_scalar_hpu_lazy(masked, scale);
  return scaled;
}

Tensor matmul_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyOp<Tensor> k(
      "aten::matmul",
      {self, other},
      {},
      {MatMulOperator::compute_output_shape(self, other)});
  return k.call();
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
          scores.options().dtype(c10::ScalarType::Int),
          scores.suggest_memory_format(),
          false);
      std::get<1>(results) = empty_hpu_lazy(
          valid_box_id_out_shape,
          scores.options().dtype(c10::ScalarType::Int),
          scores.suggest_memory_format(),
          false);
      std::get<2>(results) = empty_hpu_lazy(
          shape_tensor_shape,
          scores.options().dtype(c10::ScalarType::Int),
          scores.suggest_memory_format(),
          false);
      return results;
    }
  };

  std::vector<int64_t> box_id_out_shape{scores.sizes()[0]};
  std::vector<int64_t> valid_box_id_out_shape{1};
  std::vector<int64_t> shape_tensor_shape{5};
  HabanaNMSLazy k(
      {boxes, scores, Scalar(iou_threshold), Scalar(score_threshold)},
      {box_id_out_shape, valid_box_id_out_shape, shape_tensor_shape});
  auto result_nms = k.call();
  auto box_id_out = std::get<0>(result_nms);
  auto valid_box_id_out = std::get<1>(result_nms);
  auto shape_tensor = std::get<2>(result_nms);
  auto hl_box = GetHbLazyTensor(box_id_out);
  auto hl_valid = GetHbLazyTensor(valid_box_id_out);
  auto hl_shape = GetHbLazyTensor(shape_tensor);

  // Force an execution here to capture valid_box_id_out.
  // This element is required to determine shape of next node's output
  std::vector<HbLazyTensor> hl_flush = {hl_box, hl_valid, hl_shape};
  HbLazyTensor::SyncTensorsGraph(&hl_flush);
  auto end = valid_box_id_out.item<int64_t>();

  if (end == 0) {
    return box_id_out;
  }

  // Extract correct output using shape information.
  // Add a slice node to capture relevent elements
  auto sliced_shape = DimVector{end};
  auto node = std::make_shared<ir::Slice>(box_id_out, 0, 0, end, 1);
  auto result = empty_hpu_lazy(
      sliced_shape,
      scores.options().dtype(c10::ScalarType::Int),
      scores.suggest_memory_format(),
      false);
  auto hl_result = GetOrCreateHbLazyTensor(result, c10::kHPU);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
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
  PT_LAZY_TRACE;
  // TBD: add out_shape computation for NHWC inputs also.
  std::vector<int64_t> out_shape{
      num_rois.sizes()[0], images.sizes()[1], output_h, output_w};
  LazyOp<at::Tensor> k(
      "hpu::roi_align_fwd",
      {images,
       rois,
       num_rois,
       output_h,
       output_w,
       mode,
       sampling_ratio,
       spatial_scale,
       aligned},
      {},
      {out_shape});
  return k.call();
}

Tensor isnan_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  struct Kernel : public LazyOp<at::Tensor> {
    explicit Kernel(const Tensor& self)
        : LazyOp<at::Tensor>("aten::isnan", {self}, {}, {}, -1), m_self(self) {}

    at::Tensor get_result_overrideable() override {
      auto res = empty_hpu_lazy(
          m_self.sizes(),
          m_self.options().dtype(c10::ScalarType::Bool),
          m_self.suggest_memory_format(),
          false);
      return res;
    }

    at::Tensor m_self;
  };
  Kernel kernel{self};
  return kernel.call();
}

Tensor silu_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k("aten::silu", {self});
  return k.call();
}

Tensor silu_backward_hpu_lazy(const Tensor& grad, const Tensor& self) {
  PT_LAZY_TRACE;
  CONVERT_0D_TO_1D(self)
  CONVERT_0D_TO_1D(grad)
  LazyOp<at::Tensor> k("aten::silu_backward", {grad, self});
  CONVERT_1D_TO_0D(self, grad)
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

Tensor& linspace_out_hpu_lazy(
    const Scalar& start,
    const Scalar& end,
    c10::optional<int64_t> steps,
    Tensor& out) {
  PT_LAZY_TRACE;

  // If step value is not provided, set it 100, following
  // the CPU implementtaion....
  // pytorch-fork/aten/src/ATen/native/RangeFactories.cpp
  // Tensor& linspace_cpu_out(...
  // ...
  // const auto steps = optional_steps.value_or(100);
  int64_t step_corrected = steps.value_or(100);
  Scalar temp_end = end;

  // Handle start==end case, change the end value and
  // hence convert to start != end, by changing end variable.
  if (start.toFloat() == temp_end.toFloat()) {
    step_corrected = 1;
    auto tmp = end.toFloat();
    tmp++;
    temp_end = Scalar(tmp);
  }

  std::vector<int64_t> out_shape = {step_corrected};
  LazyOp<at::Tensor&> k(
      "aten::linspace",
      {start, temp_end, step_corrected, out},
      {},
      {out_shape});
  return k.call(out);
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

at::Tensor frobenius_norm_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  Scalar p(2.0); // p = 2.0 for Frobenius Norm
  std::vector<int64_t> shape = {};
  LazyOp<at::Tensor> k{"aten::norm", {self, p}, {}, {shape}};
  return k.call();
}

Tensor floor_divide_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  // floor_divide now do truncation:
  // https://pytorch.org/docs/stable/generated/torch.floor_divide.html
  return div(self, other, "trunc");
}

} // namespace habana_lazy
